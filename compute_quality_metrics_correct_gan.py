#!/usr/bin/env python3
"""
计算Pix2Pix GAN增强样本的质量指标

对比：
- X_real[i]: 被试1 Session 1的第i个真实样本
- X_augmented[i]: 基于X_real[i]生成的增强样本

质量指标：
1. PSD相关性 (频域)
2. 时域相关性 (时域)
3. 频率相似度 (频域)
"""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from data_loader import load_bci2a_data

def compute_psd(X, fs=250):
    """计算功率谱密度"""
    n_samples, n_channels, n_times = X.shape
    freqs, psd = signal.welch(X, fs=fs, nperseg=min(256, n_times), axis=-1)
    return freqs, psd

def compute_quality_metrics(X_real, X_augmented):
    """
    计算质量指标
    
    X_real: (N, 22, 1000) - 真实样本
    X_augmented: (N, 22, 1000) - 增强样本（基于X_real生成）
    """
    print("\n" + "="*60)
    print("计算质量指标")
    print("="*60)
    
    # 1. PSD相关性
    print("\n1. 计算PSD相关性...")
    freqs_real, psd_real = compute_psd(X_real)
    freqs_aug, psd_aug = compute_psd(X_augmented)
    
    # 对每个样本计算PSD相关性
    psd_corrs = []
    for i in range(len(X_real)):
        # 对所有通道的PSD展平
        psd_r = psd_real[i].flatten()
        psd_a = psd_aug[i].flatten()
        corr, _ = pearsonr(psd_r, psd_a)
        psd_corrs.append(corr)
    
    psd_corr_mean = np.mean(psd_corrs)
    print(f"   PSD相关性: {psd_corr_mean:.4f}")
    
    # 2. 时域相关性
    print("\n2. 计算时域相关性...")
    temporal_corrs = []
    for i in range(len(X_real)):
        # 对每个样本的所有通道展平
        x_r = X_real[i].flatten()
        x_a = X_augmented[i].flatten()
        corr, _ = pearsonr(x_r, x_a)
        temporal_corrs.append(corr)
    
    temporal_corr_mean = np.mean(temporal_corrs)
    print(f"   时域相关性: {temporal_corr_mean:.4f}")
    
    # 3. 频率相似度
    print("\n3. 计算频率相似度...")
    # Alpha频段 (8-13 Hz)
    alpha_mask = (freqs_real >= 8) & (freqs_real <= 13)
    psd_real_alpha = psd_real[:, :, alpha_mask].mean(axis=(1, 2))
    psd_aug_alpha = psd_aug[:, :, alpha_mask].mean(axis=(1, 2))
    
    # Beta频段 (13-30 Hz)
    beta_mask = (freqs_real >= 13) & (freqs_real <= 30)
    psd_real_beta = psd_real[:, :, beta_mask].mean(axis=(1, 2))
    psd_aug_beta = psd_aug[:, :, beta_mask].mean(axis=(1, 2))
    
    # 计算相似度
    alpha_corr, _ = pearsonr(psd_real_alpha, psd_aug_alpha)
    beta_corr, _ = pearsonr(psd_real_beta, psd_aug_beta)
    freq_sim = (alpha_corr + beta_corr) / 2
    
    print(f"   Alpha频段相关性: {alpha_corr:.4f}")
    print(f"   Beta频段相关性: {beta_corr:.4f}")
    print(f"   频率相似度: {freq_sim:.4f}")
    
    # 4. 平均相关性
    mean_corr = (psd_corr_mean + temporal_corr_mean + freq_sim) / 3
    print(f"\n平均相关性: {mean_corr:.4f}")
    
    return {
        'psd_corr': psd_corr_mean,
        'temporal_corr': temporal_corr_mean,
        'freq_sim': freq_sim,
        'mean_corr': mean_corr
    }

def main():
    print("="*60)
    print("Pix2Pix GAN增强样本质量评估")
    print("="*60)
    
    # 加载真实数据
    print("\n加载真实数据...")
    X, y, subjects, sessions = load_bci2a_data()
    
    # 被试1 Session 1数据
    subject1_session1_mask = (subjects == 0) & (sessions == 0)
    X_real = X[subject1_session1_mask]
    y_real = y[subject1_session1_mask]
    
    print(f"被试1 Session 1数据: {X_real.shape}")
    
    # 加载GAN增强样本
    print("\n加载GAN增强样本...")
    X_augmented_all = np.load('outputs/gan_samples/pix2pix_augmented_session1.npy')
    y_augmented_all = np.load('outputs/gan_samples/pix2pix_augmented_labels.npy')
    subjects_augmented_all = np.load('outputs/gan_samples/pix2pix_augmented_subjects.npy')
    
    # 提取被试1的增强样本
    subject1_mask = subjects_augmented_all == 0
    X_augmented = X_augmented_all[subject1_mask]
    y_augmented = y_augmented_all[subject1_mask]
    
    print(f"被试1增强样本: {X_augmented.shape}")
    
    # 验证对应关系
    assert len(X_real) == len(X_augmented), "样本数量不匹配"
    assert np.array_equal(y_real, y_augmented), "标签不匹配"
    
    print("\n✅ 数据对应关系验证通过")
    print(f"   真实样本: {X_real.shape}")
    print(f"   增强样本: {X_augmented.shape}")
    print(f"   对应关系: X_real[i] ↔ X_augmented[i]")
    
    # 计算质量指标
    metrics = compute_quality_metrics(X_real, X_augmented)
    
    # 总结
    print("\n" + "="*60)
    print("质量指标总结")
    print("="*60)
    print(f"\nPix2Pix GAN增强样本质量:")
    print(f"  PSD相关性:    {metrics['psd_corr']:.4f}")
    print(f"  时域相关性:   {metrics['temporal_corr']:.4f}")
    print(f"  频率相似度:   {metrics['freq_sim']:.4f}")
    print(f"  平均相关性:   {metrics['mean_corr']:.4f}")
    
    print("\n说明:")
    print("  - PSD相关性: 频域功率谱的相似度")
    print("  - 时域相关性: 时域波形的相似度")
    print("  - 频率相似度: Alpha/Beta频段的相似度")
    print("  - 平均相关性: 三个指标的平均值")
    
    print("\n解释:")
    print("  高相关性 (0.7-0.9) 表示增强样本保留了原始样本的特征")
    print("  这是数据增强的目标：生成相似但不完全相同的变体")

if __name__ == '__main__':
    main()
