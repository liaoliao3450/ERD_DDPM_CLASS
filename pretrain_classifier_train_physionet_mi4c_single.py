#!/usr/bin/env python3
"""
PhysioNet MI4C 四分类预训练分类器（单模型版本）

与 pretrain_classifier_train.py 结构一致，只需运行一次
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 修复导入路径（与原始脚本一致）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')

from class_discriminative import EEGClassifier, pretrain_classifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(data_dir):
    """加载数据并标准化"""
    X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
    y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
    
    # 数据标准化（按通道归一化）
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1e-6
    X = (X - mean) / std
    
    print(f"  数据标准化: mean={mean.mean():.4f}, std={std.mean():.4f}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    print("="*70)
    print("🎓 PhysioNet MI4C 预训练分类器（单模型版本）")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    print("📁 加载数据...")
    data_dir = 'data/processed/PhysioNetMI4C/PhysioNetMI4C'
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 获取数据维度
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    print(f"  通道数: {n_channels}, 采样点: {n_samples}, 类别数: {num_classes}")
    
    # 创建模型（使用与原始脚本相同的EEGClassifier）
    print("\n🎯 创建模型...")
    clf = EEGClassifier(channels=n_channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)
    
    # 预训练分类器（使用原始脚本的pretrain_classifier函数，学习率1e-3）
    print(f"\n📚 预训练分类器 ({500} epochs)...")
    clf = pretrain_classifier(
        clf,
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        epochs=500,
        batch_size=64,
        lr=1e-3,  # 使用与原始脚本相同的学习率
        device=DEVICE,
        verbose=True
    )
    
    # 测试集评估
    print("\n📊 测试集评估...")
    clf.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        outputs = clf(X_test_tensor)
        preds = torch.argmax(outputs, dim=1)
        test_acc = (preds == y_test_tensor).sum().item() / len(y_test)
    print(f"  测试集准确率: {test_acc:.4f}")
    
    # 保存分类器
    print("\n💾 保存预训练分类器...")
    os.makedirs('checkpoints', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': clf.state_dict(),
        'channels': n_channels,
        'n_samples': n_samples,
        'num_classes': num_classes,
        'test_acc': float(test_acc),
    }
    
    torch.save(checkpoint, 'checkpoints/pretrained_classifier_physionet_mi4c.pt')
    print("  ✅ 保存到: checkpoints/pretrained_classifier_physionet_mi4c.pt")
    
    # 测试加载
    print("\n🔍 测试加载...")
    clf_test = EEGClassifier(channels=n_channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)
    clf_test.load_state_dict(checkpoint['model_state_dict'])
    print("  ✅ 加载成功！")
    
    print("\n" + "="*70)
    print("✅ PhysioNet MI4C 预训练分类器已保存！")
    print("="*70)


if __name__ == '__main__':
    main()
