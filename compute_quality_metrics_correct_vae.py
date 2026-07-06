#!/usr/bin/env python3
"""
计算正确的VAE数据增强的质量指标

现在VAE是基于真实样本的变体，所以时域相关性有意义了
"""
import sys
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

sys.path.insert(0, 'utils')
from data_loader import load_bci2a_data

def compute_quality_metrics(X_real, X_augmented):
    """计算质量指标"""
    
    n_samples, n_channels, n_times = X_real.shape
    
    # 1. PSD相关性
    psd_corrs = []
    for i in range(n_samples):
        for ch in range(n_channels):
            f_real, psd_real = signal.welch(X_real[i, ch], fs=250, nperseg=min(256, n_times))
            f_aug, psd_aug = signal.welch(X_augmented[i, ch], fs=250, nperseg=min(256, n_times))
            
            if len(psd_real) > 1:
                corr, _ = pearsonr(psd_real, psd_aug)
                psd_corrs.append(corr)
    
    psd_corr = np.mean(psd_corrs)
    
    # 2. 时域相关性（现在有意义了！）
    temp_corrs = []
    for i in range(n_samples):
        for ch in range(n_channels):
            corr, _ = pearsonr(X_real[i, ch], X_augmented[i, ch])
            temp_corrs.append(corr)
    
    temp_corr = np.mean(temp_corrs)
    
    # 3. 频段相似度
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_corrs = {band: [] for band in bands}
    
    for i in range(n_samples):
        for ch in range(n_channels):
            f, psd_real = signal.welch(X_real[i, ch], fs=250, nperseg=min(256, n_times))
            _, psd_aug = signal.welch(X_augmented[i, ch], fs=250, nperseg=min(256, n_times))
            
            for band_name, (low, high) in bands.items():
                freq_mask = (f >= low) & (f <= high)
                if freq_mask.sum() > 1:
                    band_real = psd_real[freq_mask]
                    band_aug = psd_aug[freq_mask]
                    
                    if len(band_real) > 1:
                        corr, _ = pearsonr(band_real, band_aug)
                        band_corrs[band_name].append(corr)
    
    freq_sim = np.mean([np.mean(corrs) for corrs in band_corrs.values()])
    
    return {
        'psd_correlation': psd_corr,
        'temporal_correlation': temp_corr,
        'frequency_similarity': freq_sim,
        'mean_correlation': np.mean([psd_corr, temp_corr, freq_sim])
    }

def main():
    print("="*60)
    print("计算正确的VAE数据增强质量指标")
    print("="*60)
    
    # 加载数据
    X, y, subjects, sessions = load_bci2a_data()
    
    # Session 1数据
    session1_mask = sessions == 0
    X_session1 = X[session1_mask]
    
    # 加载VAE增强数据
    X_augmented = np.load('outputs/vae_samples/vae_augmented_session1.npy')
    
    print(f"\n原始数据: {X_session1.shape}")
    print(f"增强数据: {X_augmented.shape}")
    
    # 使用前100个样本进行质量评估
    n_samples = 100
    X_real_subset = X_session1[:n_samples]
    X_aug_subset = X_augmented[:n_samples]
    
    print(f"\n使用前{n_samples}个样本进行质量评估")
    
    # 计算质量指标
    print("\n计算质量指标...")
    metrics = compute_quality_metrics(X_real_subset, X_aug_subset)
    
    print("\n" + "="*60)
    print("质量指标结果")
    print("="*60)
    print(f"PSD Correlation:       {metrics['psd_correlation']:.4f}")
    print(f"Temporal Correlation:  {metrics['temporal_correlation']:.4f}")
    print(f"Frequency Similarity:  {metrics['frequency_similarity']:.4f}")
    print(f"Mean Correlation:      {metrics['mean_correlation']:.4f}")
    
    print("\n" + "="*60)
    print("说明")
    print("="*60)
    print("""
现在的时域相关性有意义了！

因为：
- 增强样本是基于真实样本的变体（编码 -> 加扰动 -> 解码）
- X_real[i] 和 X_augmented[i] 是对应的（同一个样本的增强版本）
- 时域相关性衡量的是增强样本与原始样本的相似度

预期结果：
- 时域相关性应该比较高（0.7-0.9），说明增强样本保持了原始样本的时域特征
- PSD相关性应该很高（>0.9），说明频谱特性保持得很好
- 如果时域相关性太低（<0.5），说明扰动太大
- 如果时域相关性太高（>0.95），说明扰动太小，增强效果不明显
""")

if __name__ == '__main__':
    main()
