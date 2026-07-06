#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布质量定量指标计算

计算以下指标：
1. Separation Ratio = inter-class distance / intra-class distance (↑)
2. Maximum Mean Discrepancy (MMD) (↓)
3. Fréchet Inception Distance (FID) (↓)
4. Silhouette Score (↑)

支持数据集: BCI2a, BCI2b
支持方法: Gaussian Noise, SMOTE, CVAE, WaveGAN, Cond-DDPM,
          BrainDiff, EEGDiff, DiffEEGBooth, DDPM (Ours)
"""
import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'baselines'))
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'ddpm'))
sys.path.insert(0, os.path.join(project_root, 'experiments', 'paper_experiments'))

from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE
from class_discriminative import MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
from core.classifiers.eegnet import EEGNet

# 复用 compute_generative_quality_is 中的函数
from compute_generative_quality_is import (
    load_dataset, pretrain_feature_extractor,
    load_ddpm, load_baseline, generate_samples,
    gaussian_noise_augment, smote_augment,
    EEGNetFeatureExtractor,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(project_root, 'checkpoints')
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'results', 'distribution_metrics')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 分布指标计算
# ============================================================================
def compute_separation_ratio(features, labels):
    """类别分离比 = 类间距离 / 类内距离 (越大越好)"""
    n_classes = len(np.unique(labels))

    # 类内距离（各类样本到类中心的平均距离）
    intra_class_dist = 0
    for c in range(n_classes):
        mask = labels == c
        X_c = features[mask]
        if len(X_c) > 1:
            center = X_c.mean(axis=0)
            dist = np.mean(np.linalg.norm(X_c - center, axis=1))
            intra_class_dist += dist
    intra_class_dist /= n_classes

    # 类间距离（各类中心之间的平均距离）
    centers = []
    for c in range(n_classes):
        mask = labels == c
        X_c = features[mask]
        if len(X_c) > 0:
            centers.append(X_c.mean(axis=0))

    inter_class_dist = 0
    count = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            inter_class_dist += np.linalg.norm(centers[i] - centers[j])
            count += 1
    if count > 0:
        inter_class_dist /= count

    separation_ratio = inter_class_dist / intra_class_dist if intra_class_dist > 0 else 0
    return separation_ratio, intra_class_dist, inter_class_dist


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """Maximum Mean Discrepancy (越小表示两个分布越接近)"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def rbf_kernel(X, Y, gamma):
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        XY = np.dot(X, Y.T)
        distances = XX + YY - 2 * XY
        return np.exp(-gamma * np.maximum(distances, 0))

    if kernel == 'rbf':
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
    else:
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)

    mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    return max(0, mmd)


def compute_fid(X_real, X_generated):
    """Fréchet Inception Distance (越小越好)

    使用 PCA 降维 + 协方差正则化以保证数值稳定性，与 compute_generative_quality_is.py 对齐。
    """
    from sklearn.decomposition import PCA

    # PCA 降维以稳定协方差估计
    combined = np.vstack([X_real, X_generated])
    n_components = min(64, len(X_real) - 2, len(X_generated) - 2, combined.shape[1])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)
    real_pca = combined_pca[:len(X_real)]
    gen_pca = combined_pca[len(X_real):]

    mu_real = np.mean(real_pca, axis=0)
    mu_gen = np.mean(gen_pca, axis=0)

    # 协方差正则化以保证正定性
    sigma_real = np.cov(real_pca, rowvar=False) + 0.01 * np.eye(n_components)
    sigma_gen = np.cov(gen_pca, rowvar=False) + 0.01 * np.eye(n_components)

    diff = mu_real - mu_gen
    mean_diff = np.sum(diff ** 2)

    covmean = sqrtm(sigma_real @ sigma_gen)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(max(0, fid))


def extract_features(X, feature_extractor):
    """使用预训练的EEGNet特征提取器提取特征"""
    feature_extractor.eval()
    features_list = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
            feats = feature_extractor.get_features(batch)
            features_list.append(feats.cpu().numpy())
    return np.concatenate(features_list, axis=0)


# ============================================================================
# 主流程
# ============================================================================
def evaluate_dataset(dataset, n_per_class=100):
    """评估一个数据集的所有方法"""
    print(f"\n{'=' * 80}")
    print(f"Evaluating {dataset}")
    print(f"{'=' * 80}")

    # 加载数据
    X, y, channels, n_samples, fs, num_classes, data_mean, data_std = load_dataset(dataset)

    # 预训练特征提取器
    print("\nPre-training feature extractor...")
    feature_extractor = pretrain_feature_extractor(channels, n_samples, num_classes, X, y, epochs=300)

    # 提取真实数据特征
    print("\nExtracting real features...")
    features_real = extract_features(X, feature_extractor)
    print(f"  Real features shape: {features_real.shape}")

    # 真实数据的指标
    sep_real, intra_real, inter_real = compute_separation_ratio(features_real, y)
    sil_real = silhouette_score(features_real, y)
    print(f"\nReal Data:")
    print(f"  Sep. Ratio: {sep_real:.4f}, Intra: {intra_real:.2f}, Inter: {inter_real:.2f}, Silhouette: {sil_real:.4f}")

    # 定义所有方法
    method_configs = [
        ('gaussian_noise', 'Gaussian Noise'),
        ('smote', 'SMOTE'),
        ('cvae', 'CVAE'),
        ('wavegan', 'WaveGAN'),
        ('cond_ddpm', 'Cond-DDPM'),
        ('braindiff', 'BrainDiff'),
        ('eegdiff', 'EEGDiff'),
        ('diffeegbooth', 'DiffEEGBooth'),
        ('ddpm', 'DDPM (Ours)'),
    ]

    results = {
        'Method': ['Real Data'],
        'Sep. Ratio↑': [f'{sep_real:.4f}'],
        'Intra-Dist↓': [f'{intra_real:.2f}'],
        'Inter-Dist↑': [f'{inter_real:.2f}'],
        'MMD↓': ['-'],
        'FID↓': ['-'],
        'Silhouette↑': [f'{sil_real:.4f}'],
    }

    all_gen_features = []
    all_gen_labels = []
    method_names_display = []

    for model_key, display_name in method_configs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {display_name}...")
        print(f"{'=' * 60}")

        try:
            # 生成样本
            if model_key == 'gaussian_noise':
                gen_X, gen_y = gaussian_noise_augment(X, y, n_per_class, num_classes)
            elif model_key == 'smote':
                gen_X, gen_y = smote_augment(X, y, n_per_class, num_classes)
            elif model_key == 'ddpm':
                model, (ckpt_mean, ckpt_std) = load_ddpm(dataset, channels, n_samples, fs, num_classes)
                if model is None:
                    print(f"  No DDPM checkpoint, skipping")
                    results['Method'].append(display_name)
                    for k in ['Sep. Ratio↑', 'Intra-Dist↓', 'Inter-Dist↑', 'MMD↓', 'FID↓', 'Silhouette↑']:
                        results[k].append('N/A')
                    continue
                # 保留分类器引导（论文创新），使用合理 guidance_scale
                gen_X, gen_y = generate_samples(model, 'ddpm', n_per_class, num_classes, guidance_scale=0.1)
            elif model_key in ('cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth'):
                model = load_baseline(model_key, dataset, channels, n_samples, fs, num_classes)
                if model is None:
                    print(f"  No checkpoint for {model_key}, skipping")
                    results['Method'].append(display_name)
                    for k in ['Sep. Ratio↑', 'Intra-Dist↓', 'Inter-Dist↑', 'MMD↓', 'FID↓', 'Silhouette↑']:
                        results[k].append('N/A')
                    continue
                gen_X, gen_y = generate_samples(model, model_key, n_per_class, num_classes)
            else:
                continue

            print(f"  Generated: {gen_X.shape}, range=[{gen_X.min():.2f}, {gen_X.max():.2f}]")

            # 提取特征
            features_gen = extract_features(gen_X, feature_extractor)

            # 计算指标
            sep_ratio, intra_dist, inter_dist = compute_separation_ratio(features_gen, gen_y)
            mmd = compute_mmd(features_real, features_gen)
            fid = compute_fid(features_real, features_gen)
            sil = silhouette_score(features_gen, gen_y)

            results['Method'].append(display_name)
            results['Sep. Ratio↑'].append(f'{sep_ratio:.4f}')
            results['Intra-Dist↓'].append(f'{intra_dist:.2f}')
            results['Inter-Dist↑'].append(f'{inter_dist:.2f}')
            results['MMD↓'].append(f'{mmd:.4f}')
            results['FID↓'].append(f'{fid:.2f}')
            results['Silhouette↑'].append(f'{sil:.4f}')

            print(f"  Sep. Ratio: {sep_ratio:.4f} (Real: {sep_real:.4f})")
            print(f"  Intra-Dist: {intra_dist:.2f} (Real: {intra_real:.2f})")
            print(f"  Inter-Dist: {inter_dist:.2f} (Real: {inter_real:.2f})")
            print(f"  MMD: {mmd:.4f}")
            print(f"  FID: {fid:.2f}")
            print(f"  Silhouette: {sil:.4f} (Real: {sil_real:.4f})")

            # 释放显存
            del features_gen
            if 'model' in dir():
                del model
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            results['Method'].append(display_name)
            for k in ['Sep. Ratio↑', 'Intra-Dist↓', 'Inter-Dist↑', 'MMD↓', 'FID↓', 'Silhouette↑']:
                results[k].append('N/A')

    # 打印汇总表格（紧凑格式，避免终端换行）
    print(f"\n{'=' * 78}")
    print(f"Distribution Metrics Summary - {dataset}")
    print(f"{'=' * 78}")
    # 分两段打印以避免终端换行：分离指标 + 分布指标
    print(f"\n[Separation Metrics]")
    header1 = f"{'Method':<18} {'Sep.Ratio↑':<11} {'Intra↓':<9} {'Inter↑':<9} {'Sil↑':<9}"
    print(header1)
    print("-" * 60)
    for i in range(len(results['Method'])):
        row = f"{results['Method'][i]:<18} "
        row += f"{results['Sep. Ratio↑'][i]:<11} "
        row += f"{results['Intra-Dist↓'][i]:<9} "
        row += f"{results['Inter-Dist↑'][i]:<9} "
        row += f"{results['Silhouette↑'][i]:<9}"
        print(row)

    print(f"\n[Distribution Metrics]")
    header2 = f"{'Method':<18} {'MMD↓':<11} {'FID↓':<11}"
    print(header2)
    print("-" * 42)
    for i in range(len(results['Method'])):
        row = f"{results['Method'][i]:<18} "
        row += f"{results['MMD↓'][i]:<11} "
        row += f"{results['FID↓'][i]:<11}"
        print(row)

    # 保存JSON
    json_path = os.path.join(OUTPUT_DIR, f'{dataset}_distribution_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {json_path}")

    # 保存LaTeX表格
    latex_path = os.path.join(OUTPUT_DIR, f'{dataset}_distribution_metrics.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[!t]\n")
        f.write(f"\\caption{{Distribution Quality Metrics ({dataset.upper()})}}\n")
        f.write(f"\\label{{tab:distribution_metrics_{dataset}}}\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Sep. Ratio$\\uparrow$ & Intra-Dist$\\downarrow$ & Inter-Dist$\\uparrow$ & MMD$\\downarrow$ & FID$\\downarrow$ & Silhouette$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        for i in range(len(results['Method'])):
            method = results['Method'][i]
            if method == 'DDPM (Ours)':
                method = '\\textbf{DDPM (Ours)}'
            row = f"{method} & "
            row += f"{results['Sep. Ratio↑'][i]} & "
            row += f"{results['Intra-Dist↓'][i]} & "
            row += f"{results['Inter-Dist↑'][i]} & "
            row += f"{results['MMD↓'][i]} & "
            row += f"{results['FID↓'][i]} & "
            row += f"{results['Silhouette↑'][i]} \\\\\n"
            if i == 0:
                row += "\\midrule\n"
            f.write(row)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX saved to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute distribution quality metrics')
    parser.add_argument('--dataset', type=str, default='bci2a',
                        choices=['bci2a', 'bci2b', 'physionet'],
                        help='Dataset to evaluate')
    parser.add_argument('--n_per_class', type=int, default=100,
                        help='Number of samples to generate per class')
    args = parser.parse_args()

    evaluate_dataset(args.dataset, args.n_per_class)


if __name__ == '__main__':
    main()
