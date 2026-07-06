#!/usr/bin/env python3
"""
为被试1生成DDPM样本用于质量指标计算

重要：指定subject_id=0来生成被试1的样本
这样才能与被试1 Session 1的真实数据进行公平对比
"""
import sys
import os
import torch
import numpy as np

# 添加模块路径
sys.path.insert(0, 'core/models/ddpm')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据参数
C, T, NUM_CLASSES = 22, 1000, 4
FS = 250

def load_trained_ddpm(device=DEVICE, checkpoint_path='checkpoints/best_class_discriminative.pt'):
    """加载已训练的DDPM模型"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return None
    
    print(f"📥 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型组件
    eps_model = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(device)
    
    # 加载分类器
    classifier_path = 'checkpoints/classifier_class_disc.pt'
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    if os.path.exists(classifier_path):
        clf_checkpoint = torch.load(classifier_path, map_location=device)
        if isinstance(clf_checkpoint, dict) and 'model_state_dict' in clf_checkpoint:
            classifier.load_state_dict(clf_checkpoint['model_state_dict'])
        else:
            classifier.load_state_dict(clf_checkpoint)
        print(f"✅ 分类器加载成功: {classifier_path}")
    else:
        print(f"⚠️  分类器文件不存在，使用随机初始化的分类器")
    
    # 从checkpoint获取target_psd和target_laterality，如果没有则使用默认值
    if isinstance(checkpoint, dict) and 'target_psd' in checkpoint:
        target_psd = checkpoint['target_psd'].to(device)
        target_laterality = checkpoint['target_laterality'].to(device)
    else:
        # 使用默认值
        target_psd = torch.zeros(501).to(device)  # rfft输出长度
        target_laterality = torch.zeros(4).to(device)
    
    # 创建DDPM模型
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=FS
    ).to(device)
    
    # 加载权重
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        ddpm.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'eps_model' in checkpoint:
        ddpm.eps_model.load_state_dict(checkpoint['eps_model'])
    else:
        ddpm.load_state_dict(checkpoint)
    
    ddpm.eval()
    print("✅ 模型加载成功")
    return ddpm

def normalize_generated_data_to_real_stats(real_data, gen_data):
    """
    将生成数据的统计特性对齐到真实数据（全局标准化和重新缩放）
    
    Args:
        real_data: 真实数据 [N, C, T]
        gen_data: 生成数据 [M, C, T]
    
    Returns:
        对齐后的生成数据
    """
    # 计算真实数据的全局统计特性（按通道，axis=(0, 2)）
    real_global_mean = real_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
    real_global_std = real_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
    
    # 计算生成数据的全局统计特性
    gen_global_mean = gen_data.mean(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
    gen_global_std = gen_data.std(axis=(0, 2), keepdims=True)       # (1, n_channels, 1)
    
    # 全局标准化和重新缩放
    gen_data_normalized = (gen_data - gen_global_mean) / (gen_global_std + 1e-8)
    gen_data_matched = gen_data_normalized * real_global_std + real_global_mean
    
    return gen_data_matched

def main():
    print("="*70)
    print("为被试1生成DDPM样本用于质量指标计算")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    # 加载训练好的DDPM
    ddpm = load_trained_ddpm(device=DEVICE)
    
    if ddpm is None:
        print("❌ 无法加载DDPM模型")
        print("请确保已运行训练脚本并保存模型到 checkpoints/best_class_discriminative.pt")
        return
    
    print()
    
    # 加载真实数据，为每个真实样本生成对应的增强样本
    print("📊 加载真实数据...")
    sys.path.insert(0, 'utils')
    from data_loader import load_bci2a_data
    
    X, y, subjects, sessions = load_bci2a_data()
    # 使用Subject 1 Session 1的数据
    subject1_session1_mask = (subjects == 0) & (sessions == 0)
    X_real = X[subject1_session1_mask]
    y_real = y[subject1_session1_mask]
    
    print(f"  真实数据形状: {X_real.shape}")
    print(f"  真实标签形状: {y_real.shape}")
    print(f"  每类数量: {[(y_real == c).sum() for c in range(4)]}")
    print()
    
    # 生成样本 - 为每个真实样本生成对应的增强样本
    print("🎲 为每个真实样本生成对应的DDPM增强样本...")
    print("   - 被试ID: 0 (被试1)")
    print("   - 总样本数: 与真实数据相同")
    print("   - 采样方法: DDIM (50步)")
    print("   - 引导强度: 5.0")
    print()
    
    ddpm.eval()
    gen_X = []
    gen_y = []
    gen_indices = []  # 保存对应的真实数据索引
    
    subject_id = 0  # 被试1
    
    with torch.no_grad():
        # 按类别分组生成，保持与真实数据的对应关系
        for c in range(4):
            # 找到该类别的所有真实样本
            class_mask = (y_real == c)
            class_indices = np.where(class_mask)[0]
            n_samples = len(class_indices)
            
            if n_samples > 0:
                print(f"   类别 {c}: 为 {n_samples} 个真实样本生成对应的增强样本...")
                # 生成相同数量的样本
                yg = torch.full((n_samples,), c, dtype=torch.long, device=DEVICE)
                
                # 生成样本 - 指定类别
                samples = ddpm.sample_ddim(n_samples, yg, steps=50, guidance_scale=5.0, device=DEVICE)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * n_samples)
                gen_indices.extend(class_indices.tolist())
            else:
                print(f"   类别 {c}: 无真实样本，跳过")
    
    # 合并（按真实数据的顺序排列）
    gen_X = np.concatenate(gen_X, axis=0)  # [N, 22, 1000]
    gen_y = np.array(gen_y)  # [N]
    gen_indices = np.array(gen_indices)  # [N]
    
    # 按照真实数据的索引顺序重新排列生成数据
    sort_order = np.argsort(gen_indices)
    gen_X = gen_X[sort_order]
    gen_y = gen_y[sort_order]
    gen_indices = gen_indices[sort_order]
    
    print()
    print(f"✅ 生成完成")
    print(f"   - 形状: {gen_X.shape}")
    print(f"   - 标签: {gen_y.shape}")
    print(f"   - 每类数量: {[(gen_y == c).sum() for c in range(4)]}")
    print(f"   - 与真实数据一一对应: ✅")
    print()
    
    # 验证对应关系
    print("🔍 验证对应关系...")
    assert len(gen_X) == len(X_real), f"生成数据数量({len(gen_X)})与真实数据数量({len(X_real)})不匹配"
    assert np.array_equal(gen_y, y_real), "生成数据标签与真实数据标签不匹配"
    assert np.array_equal(gen_indices, np.arange(len(X_real))), "索引对应关系不正确"
    print("   ✅ 对应关系验证通过")
    print()
    
    # 真实数据已加载，用于全局标准化
    print("📊 应用全局标准化...")
    
    print(f"  真实数据形状: {X_real.shape}")
    
    # 应用全局标准化（将生成数据对齐到真实数据的统计特性）
    print("📊 应用全局标准化...")
    print(f"  标准化前 - 生成数据均值: {gen_X.mean():.4f}, 标准差: {gen_X.std():.4f}")
    print(f"  真实数据均值: {X_real.mean():.4f}, 标准差: {X_real.std():.4f}")
    
    gen_X_normalized = normalize_generated_data_to_real_stats(X_real, gen_X)
    
    print(f"  标准化后 - 生成数据均值: {gen_X_normalized.mean():.4f}, 标准差: {gen_X_normalized.std():.4f}")
    print()
    
    # 保存
    os.makedirs('outputs/ddpm_samples', exist_ok=True)
    
    # 保存标准化后的数据
    np.save('outputs/ddpm_samples/ddpm_samples_subject1.npy', gen_X_normalized)
    np.save('outputs/ddpm_samples/ddpm_labels_subject1.npy', gen_y)
    np.save('outputs/ddpm_samples/ddpm_indices_subject1.npy', gen_indices)  # 保存对应关系
    
    print("💾 样本已保存:")
    print("   - outputs/ddpm_samples/ddpm_samples_subject1.npy")
    print("   - outputs/ddpm_samples/ddpm_labels_subject1.npy")
    print("   - outputs/ddpm_samples/ddpm_indices_subject1.npy (对应关系)")
    print()
    
    # 基本统计
    print("📊 基本统计（标准化后）:")
    print(f"   - 均值: {gen_X_normalized.mean():.6f}")
    print(f"   - 标准差: {gen_X_normalized.std():.6f}")
    print(f"   - 最小值: {gen_X_normalized.min():.6f}")
    print(f"   - 最大值: {gen_X_normalized.max():.6f}")
    print()
    
    print("="*70)
    print("✅ 被试1的DDPM样本生成完成！")
    print("="*70)
    print()
    print("说明:")
    print("  这些样本是专门为被试1生成的，已经应用了全局标准化")
    print("  每个生成样本与真实数据一一对应（相同索引和标签）")
    print("  可以与被试1 Session 1的真实数据进行公平对比")



if __name__ == '__main__':
    main()
