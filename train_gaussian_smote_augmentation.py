#!/usr/bin/env python3
"""
传统数据增强方法：Gaussian Noise 和 SMOTE

训练：基于所有被试Session 1数据
增强：对真实样本进行增强
"""
import sys
import os
from pathlib import Path
import numpy as np
from imblearn.over_sampling import SMOTE

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from data_loader import load_bci2a_data

def augment_with_gaussian_noise(X, y, noise_scale=0.1):
    """
    用Gaussian Noise对真实样本进行增强
    
    方法：X_augmented = X_real + Gaussian_noise
    """
    print("\n" + "="*60)
    print("用Gaussian Noise进行数据增强")
    print("="*60)
    
    # 对每个样本添加高斯噪声
    noise = np.random.randn(*X.shape) * noise_scale
    X_augmented = X + noise
    
    print(f"  原始样本: {X.shape}")
    print(f"  增强样本: {X_augmented.shape}")
    print(f"  噪声强度: {noise_scale}")
    print(f"  数据范围: [{X_augmented.min():.3f}, {X_augmented.max():.3f}]")
    print(f"  均值: {X_augmented.mean():.3f}, 标准差: {X_augmented.std():.3f}")
    
    return X_augmented

def augment_with_smote(X, y):
    """
    用SMOTE对真实样本进行增强
    
    方法：在特征空间中对样本进行插值
    注意：由于数据已经平衡，我们手动实现插值逻辑
    """
    print("\n" + "="*60)
    print("用SMOTE进行数据增强")
    print("="*60)
    
    n_samples, n_channels, n_times = X.shape
    
    print(f"  原始样本: {X.shape}")
    
    # 计算每个类别的样本数
    unique, counts = np.unique(y, return_counts=True)
    print(f"  类别分布: {dict(zip(unique, counts))}")
    
    # 手动实现SMOTE插值
    # 对每个样本，找到同类别的k个最近邻，随机选择一个进行插值
    X_augmented = []
    y_augmented = []
    
    for c in unique:
        mask = y == c
        X_class = X[mask]
        n_class = len(X_class)
        
        print(f"  处理类别 {c}: {n_class} 个样本")
        
        # 对每个样本
        for i in range(n_class):
            # 随机选择另一个样本（不是自己）
            candidates = list(range(n_class))
            candidates.remove(i)
            j = np.random.choice(candidates)
            
            # 随机插值系数 (0.3-0.7之间，避免太接近原样本)
            alpha = np.random.uniform(0.3, 0.7)
            
            # 插值生成新样本
            new_sample = alpha * X_class[i] + (1 - alpha) * X_class[j]
            X_augmented.append(new_sample)
            y_augmented.append(c)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"  增强样本: {X_augmented.shape}")
    print(f"  数据范围: [{X_augmented.min():.3f}, {X_augmented.max():.3f}]")
    print(f"  均值: {X_augmented.mean():.3f}, 标准差: {X_augmented.std():.3f}")
    
    # 验证不是原样本
    is_same = np.allclose(X_augmented, X)
    print(f"  与原样本相同: {is_same}")
    if is_same:
        print("  警告: 生成的样本与原样本相同！")
    else:
        print("  成功生成不同的增强样本")
    
    return X_augmented, y_augmented

def main():
    print("="*60)
    print("传统数据增强方法：Gaussian Noise 和 SMOTE")
    print("="*60)
    
    # 加载数据
    print("\n加载BCI数据...")
    X, y, subjects, sessions = load_bci2a_data()
    
    # 所有被试的Session 1数据
    session1_mask = sessions == 0
    X_session1 = X[session1_mask]
    y_session1 = y[session1_mask]
    subjects_session1 = subjects[session1_mask]
    
    print(f"Session 1数据: {X_session1.shape}")
    print(f"类别分布: {np.bincount(y_session1)}")
    print()
    
    # ==================== Gaussian Noise增强 ====================
    print("="*60)
    print("1/2: Gaussian Noise增强")
    print("="*60)
    
    X_gaussian = augment_with_gaussian_noise(X_session1, y_session1, noise_scale=0.1)
    
    # 保存增强样本
    os.makedirs('outputs/gaussian_samples', exist_ok=True)
    np.save('outputs/gaussian_samples/gaussian_augmented_session1.npy', X_gaussian)
    np.save('outputs/gaussian_samples/gaussian_augmented_labels.npy', y_session1)
    np.save('outputs/gaussian_samples/gaussian_augmented_subjects.npy', subjects_session1)
    
    print("\nGaussian Noise增强样本已保存")
    print(f"  样本: outputs/gaussian_samples/gaussian_augmented_session1.npy")
    print(f"  标签: outputs/gaussian_samples/gaussian_augmented_labels.npy")
    print(f"  被试: outputs/gaussian_samples/gaussian_augmented_subjects.npy")

    # 额外：为t-SNE最终可视化保存Gaussian Noise缓存
    tsne_cache_dir = 'outputs/figures/tsne/cached_data'
    os.makedirs(tsne_cache_dir, exist_ok=True)
    gaussian_cache_path = os.path.join(tsne_cache_dir, 'gaussian_noise_data.npz')
    np.savez(gaussian_cache_path, X=X_gaussian, y=y_session1)
    print(f"  t-SNE缓存(Gaussian): {gaussian_cache_path}")
    
    # ==================== SMOTE增强 ====================
    print("\n" + "="*60)
    print("2/2: SMOTE增强")
    print("="*60)
    
    X_smote, y_smote = augment_with_smote(X_session1, y_session1)
    
    # 保存增强样本
    os.makedirs('outputs/smote_samples', exist_ok=True)
    np.save('outputs/smote_samples/smote_augmented_session1.npy', X_smote)
    np.save('outputs/smote_samples/smote_augmented_labels.npy', y_smote)
    np.save('outputs/smote_samples/smote_augmented_subjects.npy', subjects_session1)
    
    print("\nSMOTE增强样本已保存")
    print(f"  样本: outputs/smote_samples/smote_augmented_session1.npy")
    print(f"  标签: outputs/smote_samples/smote_augmented_labels.npy")
    print(f"  被试: outputs/smote_samples/smote_augmented_subjects.npy")

    # 额外：为t-SNE最终可视化保存SMOTE缓存
    smote_cache_path = os.path.join(tsne_cache_dir, 'smote_data.npz')
    np.savez(smote_cache_path, X=X_smote, y=y_smote)
    print(f"  t-SNE缓存(SMOTE): {smote_cache_path}")
    
    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    print(f"\n原始数据: {X_session1.shape}")
    print(f"Gaussian增强: {X_gaussian.shape}")
    print(f"SMOTE增强: {X_smote.shape}")
    
    print("\n说明:")
    print("  1. Gaussian Noise: 对每个样本添加高斯噪声")
    print("  2. SMOTE: 在特征空间中对样本进行插值")
    print("  3. 两种方法都基于真实样本进行增强")
    print("  4. 增强样本与原始样本一一对应")
    
    print("\n下一步: 计算质量指标，对比所有方法")

if __name__ == '__main__':
    main()
