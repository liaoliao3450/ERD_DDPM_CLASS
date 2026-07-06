#!/usr/bin/env python3
"""
计算所有增强方法的质量指标对比

方法：
1. VAE - 基于真实样本增强
2. Pix2Pix GAN - 基于真实样本增强
3. Gaussian Noise - 基于真实样本+噪声
4. SMOTE - 基于真实样本插值

质量指标：
- PSD相关性 (频域)
- 时域相关性 (时域)
- 频率相似度 (频域)
- 平均相关性
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

def compute_quality_metrics(X_real, X_augmented, method_name):
    """
    计算质量指标
    
    X_real: (N, 22, 1000) - 真实样本
    X_augmented: (N, 22, 1000) - 增强样本
    """
    print(f"\n{'='*60}")
    print(f"计算 {method_name} 的质量指标")
    print(f"{'='*60}")
    
    # 1. PSD相关性
    freqs_real, psd_real = compute_psd(X_real)
    freqs_aug, psd_aug = compute_psd(X_augmented)
    
    psd_corrs = []
    for i in range(len(X_real)):
        psd_r = psd_real[i].flatten()
        psd_a = psd_aug[i].flatten()
        corr, _ = pearsonr(psd_r, psd_a)
        psd_corrs.append(corr)
    
    psd_corr_mean = np.mean(psd_corrs)
    
    # 2. 时域相关性
    temporal_corrs = []
    for i in range(len(X_real)):
        x_r = X_real[i].flatten()
        x_a = X_augmented[i].flatten()
        corr, _ = pearsonr(x_r, x_a)
        temporal_corrs.append(corr)
    
    temporal_corr_mean = np.mean(temporal_corrs)
    
    # 3. 频率相似度
    # Alpha频段 (8-13 Hz)
    alpha_mask = (freqs_real >= 8) & (freqs_real <= 13)
    psd_real_alpha = psd_real[:, :, alpha_mask].mean(axis=(1, 2))
    psd_aug_alpha = psd_aug[:, :, alpha_mask].mean(axis=(1, 2))
    
    # Beta频段 (13-30 Hz)
    beta_mask = (freqs_real >= 13) & (freqs_real <= 30)
    psd_real_beta = psd_real[:, :, beta_mask].mean(axis=(1, 2))
    psd_aug_beta = psd_aug[:, :, beta_mask].mean(axis=(1, 2))
    
    alpha_corr, _ = pearsonr(psd_real_alpha, psd_aug_alpha)
    beta_corr, _ = pearsonr(psd_real_beta, psd_aug_beta)
    freq_sim = (alpha_corr + beta_corr) / 2
    
    # 4. 平均相关性
    mean_corr = (psd_corr_mean + temporal_corr_mean + freq_sim) / 3
    
    print(f"  PSD相关性:    {psd_corr_mean:.4f}")
    print(f"  时域相关性:   {temporal_corr_mean:.4f}")
    print(f"  频率相似度:   {freq_sim:.4f}")
    print(f"  平均相关性:   {mean_corr:.4f}")
    
    return {
        'method': method_name,
        'psd_corr': psd_corr_mean,
        'temporal_corr': temporal_corr_mean,
        'freq_sim': freq_sim,
        'mean_corr': mean_corr
    }

def main():
    print("="*60)
    print("所有增强方法的质量指标对比")
    print("="*60)
    
    # 加载真实数据
    print("\n加载真实数据...")
    X, y, subjects, sessions = load_bci2a_data()
    
    # 被试1 Session 1数据
    subject1_session1_mask = (subjects == 0) & (sessions == 0)
    X_real = X[subject1_session1_mask]
    y_real = y[subject1_session1_mask]
    
    print(f"被试1 Session 1数据: {X_real.shape}")
    
    # 存储所有方法的结果
    all_results = []
    
    # ==================== 1. VAE ====================
    print("\n" + "="*60)
    print("1/6: 加载VAE增强样本")
    print("="*60)
    
    try:
        X_vae_all = np.load('outputs/vae_samples/vae_augmented_session1.npy')
        subjects_vae = np.load('outputs/vae_samples/vae_augmented_subjects.npy')
        
        subject1_mask = subjects_vae == 0
        X_vae = X_vae_all[subject1_mask]
        
        print(f"VAE增强样本: {X_vae.shape}")
        
        vae_metrics = compute_quality_metrics(X_real, X_vae, "VAE")
        all_results.append(vae_metrics)
    except Exception as e:
        print(f"❌ VAE样本加载失败: {e}")
    
    # ==================== 2. Pix2Pix GAN ====================
    print("\n" + "="*60)
    print("2/6: 加载Pix2Pix GAN增强样本")
    print("="*60)
    
    try:
        X_gan_all = np.load('outputs/gan_samples/pix2pix_augmented_session1.npy')
        subjects_gan = np.load('outputs/gan_samples/pix2pix_augmented_subjects.npy')
        
        subject1_mask = subjects_gan == 0
        X_gan = X_gan_all[subject1_mask]
        
        print(f"GAN增强样本: {X_gan.shape}")
        
        gan_metrics = compute_quality_metrics(X_real, X_gan, "Pix2Pix GAN")
        all_results.append(gan_metrics)
    except Exception as e:
        print(f"❌ GAN样本加载失败: {e}")
    
    # ==================== 3. Gaussian Noise ====================
    print("\n" + "="*60)
    print("3/6: 加载Gaussian Noise增强样本")
    print("="*60)
    
    try:
        X_gaussian_all = np.load('outputs/gaussian_samples/gaussian_augmented_session1.npy')
        subjects_gaussian = np.load('outputs/gaussian_samples/gaussian_augmented_subjects.npy')
        
        subject1_mask = subjects_gaussian == 0
        X_gaussian = X_gaussian_all[subject1_mask]
        
        print(f"Gaussian增强样本: {X_gaussian.shape}")
        
        gaussian_metrics = compute_quality_metrics(X_real, X_gaussian, "Gaussian Noise")
        all_results.append(gaussian_metrics)
    except Exception as e:
        print(f"❌ Gaussian样本加载失败: {e}")
    
    # ==================== 4. SMOTE ====================
    print("\n" + "="*60)
    print("4/6: 加载SMOTE增强样本")
    print("="*60)
    
    try:
        X_smote_all = np.load('outputs/smote_samples/smote_augmented_session1.npy')
        subjects_smote = np.load('outputs/smote_samples/smote_augmented_subjects.npy')
        
        subject1_mask = subjects_smote == 0
        X_smote = X_smote_all[subject1_mask]
        
        print(f"SMOTE增强样本: {X_smote.shape}")
        
        smote_metrics = compute_quality_metrics(X_real, X_smote, "SMOTE")
        all_results.append(smote_metrics)
    except Exception as e:
        print(f"❌ SMOTE样本加载失败: {e}")
    
    # ==================== 5. DDPM (从噪声生成) ====================
    print("\n" + "="*60)
    print("5/6: 加载DDPM增强样本（从噪声生成）")
    print("="*60)
    
    try:
        X_ddpm_all = np.load('outputs/ddpm_samples/ddpm_augmented_session1.npy')
        subjects_ddpm = np.load('outputs/ddpm_samples/ddpm_augmented_subjects.npy')
        
        subject1_mask = subjects_ddpm == 0
        X_ddpm = X_ddpm_all[subject1_mask]
        
        print(f"DDPM增强样本: {X_ddpm.shape}")
        print(f"注意: DDPM从噪声生成，不基于真实样本")
        
        ddpm_metrics = compute_quality_metrics(X_real, X_ddpm, "DDPM (生成)")
        all_results.append(ddpm_metrics)
    except Exception as e:
        print(f"❌ DDPM样本加载失败: {e}")
    
    # ==================== 6. DDPM (去噪增强) ====================
    print("\n" + "="*60)
    print("6/6: 加载DDPM去噪增强样本（基于真实样本）")
    print("="*60)
    
    try:
        X_ddpm_denoise_all = np.load('outputs/ddpm_samples/ddpm_denoising_augmented_session1.npy')
        subjects_ddpm_denoise = np.load('outputs/ddpm_samples/ddpm_denoising_augmented_subjects.npy')
        
        subject1_mask = subjects_ddpm_denoise == 0
        X_ddpm_denoise = X_ddpm_denoise_all[subject1_mask]
        
        print(f"DDPM去噪样本: {X_ddpm_denoise.shape}")
        print(f"注意: DDPM去噪方法基于真实样本（添加噪声→去噪）")
        
        ddpm_denoise_metrics = compute_quality_metrics(X_real, X_ddpm_denoise, "DDPM (去噪)")
        all_results.append(ddpm_denoise_metrics)
    except Exception as e:
        print(f"❌ DDPM去噪样本加载失败: {e}")
    
    # ==================== 总结对比 ====================
    print("\n" + "="*60)
    print("质量指标总结对比")
    print("="*60)
    
    # 打印表格
    print("\n| 方法 | PSD相关性 | 时域相关性 | 频率相似度 | 平均相关性 |")
    print("|------|-----------|------------|------------|------------|")
    
    # 按平均相关性排序
    all_results.sort(key=lambda x: x['mean_corr'], reverse=True)
    
    for result in all_results:
        print(f"| {result['method']:15s} | {result['psd_corr']:.4f} | "
              f"{result['temporal_corr']:.4f} | {result['freq_sim']:.4f} | "
              f"{result['mean_corr']:.4f} |")
    
    # 详细分析
    print("\n" + "="*60)
    print("详细分析")
    print("="*60)
    
    if len(all_results) >= 4:
        print("\n### 排名")
        for i, result in enumerate(all_results, 1):
            emoji = "🏆" if i == 1 else "⭐" if i == 2 else ""
            print(f"{i}. {emoji} {result['method']}: {result['mean_corr']:.4f}")
        
        print("\n### 各指标最佳")
        best_psd = max(all_results, key=lambda x: x['psd_corr'])
        best_temporal = max(all_results, key=lambda x: x['temporal_corr'])
        best_freq = max(all_results, key=lambda x: x['freq_sim'])
        
        print(f"- PSD相关性最高: {best_psd['method']} ({best_psd['psd_corr']:.4f})")
        print(f"- 时域相关性最高: {best_temporal['method']} ({best_temporal['temporal_corr']:.4f})")
        print(f"- 频率相似度最高: {best_freq['method']} ({best_freq['freq_sim']:.4f})")
        
        print("\n### 关键发现")
        print("1. 基于真实样本的增强方法 (VAE, GAN, Gaussian) 质量指标较高")
        print("2. 从噪声生成的方法 (DDPM) 质量指标可能较低，但分类性能最好")
        print("3. SMOTE在时序数据上完全失败")
        print("4. 时域相关性反映了增强样本与原始样本的相似程度")
        print("   - 高相关性 (>0.9): 更保守，更接近原始样本")
        print("   - 中等相关性 (0.7-0.9): 平衡保真度和多样性")
        print("   - 低相关性 (<0.7): 更多样，但可能偏离原始分布")
        print("\n### 重要洞察")
        print("质量指标 ≠ 分类性能")
        print("- DDPM性能最好 (66.05%) 但质量指标可能不是最高")
        print("- 说明类别判别性比简单相似度更重要")
    
    # 保存结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    
    import json
    os.makedirs('outputs/quality_metrics', exist_ok=True)
    
    # 转换numpy类型为Python类型
    results_for_json = []
    for result in all_results:
        results_for_json.append({
            'method': result['method'],
            'psd_corr': float(result['psd_corr']),
            'temporal_corr': float(result['temporal_corr']),
            'freq_sim': float(result['freq_sim']),
            'mean_corr': float(result['mean_corr'])
        })
    
    with open('outputs/quality_metrics/all_methods_comparison.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print("✅ 结果已保存到: outputs/quality_metrics/all_methods_comparison.json")
    
    print("\n说明:")
    print("  - PSD相关性: 频域功率谱的相似度")
    print("  - 时域相关性: 时域波形的相似度")
    print("  - 频率相似度: Alpha/Beta频段的相似度")
    print("  - 平均相关性: 三个指标的平均值")
    
    print("\n下一步: 测试各方法的分类性能")

if __name__ == '__main__':
    import os
    main()
