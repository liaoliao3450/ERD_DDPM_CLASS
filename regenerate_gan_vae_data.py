#!/usr/bin/env python3
"""
使用修复后的GAN和VAE模型重新生成数据
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, '.')
from utils.data_loader import load_bci2a_data, get_subject_session_data
from core.models.timegan.model import TimeGAN
from core.models.latent_diffusion.model import LatentDiffusion1D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_gan_data(n_samples_per_class=72):
    """使用修复后的GAN生成数据"""
    print("\n" + "="*70)
    print("Generating data with fixed GAN")
    print("="*70)
    
    # 加载模型
    model = TimeGAN(channels=22, length=1000, hidden=64, z_dim=64).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/gan/gan_fixed_scale.pt', map_location=DEVICE))
    model.eval()
    
    print(f"  Loaded model with output_scale: {model.recovery.output_scale.item():.4f}")
    
    # 生成数据
    generated_samples = []
    generated_labels = []
    
    with torch.no_grad():
        for class_id in range(4):
            print(f"  Generating class {class_id}...")
            z = torch.randn(n_samples_per_class, model.hidden, model.length, device=DEVICE)
            h_fake = model.generator(z)
            x_gen = model.recovery(h_fake)
            
            generated_samples.append(x_gen.cpu().numpy())
            generated_labels.extend([class_id] * n_samples_per_class)
    
    X_gen = np.concatenate(generated_samples, axis=0)
    y_gen = np.array(generated_labels)
    
    print(f"\n  Generated data shape: {X_gen.shape}")
    print(f"  Mean: {X_gen.mean():.6f}")
    print(f"  Std: {X_gen.std():.6f}")
    print(f"  Range: [{X_gen.min():.6f}, {X_gen.max():.6f}]")
    
    return X_gen, y_gen


def generate_vae_data(n_samples_per_class=72):
    """使用修复后的VAE生成数据"""
    print("\n" + "="*70)
    print("Generating data with fixed VAE")
    print("="*70)
    
    # 加载模型
    model = LatentDiffusion1D(channels=22, z_channels=64, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/vae/vae_fixed_scale.pt', map_location=DEVICE))
    model.eval()
    
    print(f"  Loaded model with output_scale: {model.dec.output_scale.item():.4f}")
    
    # 生成数据
    generated_samples = []
    generated_labels = []
    
    with torch.no_grad():
        for class_id in range(4):
            print(f"  Generating class {class_id}...")
            z = torch.randn(n_samples_per_class, 64, device=DEVICE)
            z = z.unsqueeze(-1).repeat(1, 1, 125)  # (batch, 64, 125)
            x_gen = model.dec(z)
            
            generated_samples.append(x_gen.cpu().numpy())
            generated_labels.extend([class_id] * n_samples_per_class)
    
    X_gen = np.concatenate(generated_samples, axis=0)
    y_gen = np.array(generated_labels)
    
    print(f"\n  Generated data shape: {X_gen.shape}")
    print(f"  Mean: {X_gen.mean():.6f}")
    print(f"  Std: {X_gen.std():.6f}")
    print(f"  Range: [{X_gen.min():.6f}, {X_gen.max():.6f}]")
    
    return X_gen, y_gen


def main():
    print("="*70)
    print("重新生成GAN和VAE数据（使用修复后的模型）")
    print("="*70)
    
    # 加载原始数据用于对比
    print("\n加载原始数据...")
    X, y, subjects, sessions = load_bci2a_data()
    X_original, y_original = get_subject_session_data(X, y, subjects, sessions, 0, 0)
    
    print(f"原始数据统计:")
    print(f"  形状: {X_original.shape}")
    print(f"  均值: {X_original.mean():.6f}")
    print(f"  标准差: {X_original.std():.6f}")
    print(f"  范围: [{X_original.min():.6f}, {X_original.max():.6f}]")
    
    # 生成GAN数据
    X_gan, y_gan = generate_gan_data(n_samples_per_class=72)
    
    # 生成VAE数据
    X_vae, y_vae = generate_vae_data(n_samples_per_class=72)
    
    # 保存到缓存
    cache_dir = 'outputs/figures/tsne/cached_data'
    os.makedirs(cache_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("保存到缓存...")
    print("="*70)
    
    # 保存GAN数据
    gan_cache = os.path.join(cache_dir, 'gan_data.npz')
    np.savez(gan_cache, X=X_gan, y=y_gan)
    print(f"  ✅ GAN数据已保存: {gan_cache}")
    
    # 保存VAE数据
    vae_cache = os.path.join(cache_dir, 'vae_data.npz')
    np.savez(vae_cache, X=X_vae, y=y_vae)
    print(f"  ✅ VAE数据已保存: {vae_cache}")
    
    # 对比统计
    print("\n" + "="*70)
    print("数据质量对比")
    print("="*70)
    
    original_std = X_original.std()
    
    print(f"\n原始数据:")
    print(f"  标准差: {original_std:.6f}")
    
    print(f"\nGAN生成数据:")
    print(f"  标准差: {X_gan.std():.6f}")
    print(f"  比率: {X_gan.std() / original_std:.4f}")
    if abs(X_gan.std() - original_std) / original_std < 0.05:
        print(f"  ✅ 标准差匹配良好！")
    else:
        print(f"  ⚠️ 标准差仍有差异")
    
    print(f"\nVAE生成数据:")
    print(f"  标准差: {X_vae.std():.6f}")
    print(f"  比率: {X_vae.std() / original_std:.4f}")
    if abs(X_vae.std() - original_std) / original_std < 0.05:
        print(f"  ✅ 标准差匹配良好！")
    else:
        print(f"  ⚠️ 标准差仍有差异")
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)
    print("\n现在可以运行可视化脚本了:")
    print("  python experiments/visualize_clustering_eegnet.py")


if __name__ == '__main__':
    main()
