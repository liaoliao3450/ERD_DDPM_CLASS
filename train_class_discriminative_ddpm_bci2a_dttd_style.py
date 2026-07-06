#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCI2a: DTTD-style training — NO normalization, raw data

与原始 train_class_discriminative_ddpm.py 的关键区别:
1. 不做逐通道归一化 — 直接使用原始数据 (与DTTD一致)
2. 数据缩放: 原始数据 * 1e6 (微伏级别, 与DTTD一致)
3. 保存到 checkpoints/bci2a_dttd_style/ (不覆盖原始模型)
4. checkpoint 中不保存 data_mean/data_std (因为没有归一化)

评估时需配合 evaluate_Class_Discriminativ_ddpm_bci2a_dttd_style.py 使用
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal

sys.path.insert(0, 'core/models/ddpm')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    pretrain_classifier
)

# ============================================================================
# 配置
# ============================================================================
DATA_DIR = 'data/processed/BCI2a'
CHECKPOINT_DIR = 'checkpoints/bci2a_dttd_style'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

C, T, NUM_CLASSES = 22, 1000, 4
FS = 250
C3_IDX, C4_IDX = 7, 11

EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

NOISE_WEIGHT = 1.0
SPECTRAL_WEIGHT = 1.0
ERD_WEIGHT = 10.0
CLS_WEIGHT = 5.0

GUIDANCE_SCALE = 2.0
CLASSIFIER_PRETRAIN_EPOCHS = 50

# DTTD-style: 数据缩放因子 (原始数据 * 1e6 = 微伏级别)
DATA_SCALE = 1e6


# ============================================================================
# 数据处理 — 不做归一化，只做缩放
# ============================================================================
def load_data():
    """加载BCI2a数据 — DTTD风格: 不归一化，仅缩放到微伏级别"""
    print("加载数据 (DTTD style: no normalization, scale to uV)...")

    X = np.load(f'{DATA_DIR}/X.npy').astype(np.float32) * DATA_SCALE  # 缩放到微伏
    y = np.load(f'{DATA_DIR}/y.npy')

    # 使用session 0作为训练数据
    sess_ids = np.tile(np.repeat([0, 1], 288), 9)
    mask = sess_ids == 0
    X_train = X[mask]
    y_train = (y[mask] - 1).astype(np.int64)

    # DTTD风格: 不做逐通道归一化
    print(f"  训练数据: {X_train.shape}")
    print(f"  类别分布: {np.bincount(y_train)}")
    print(f"  数据范围: [{X_train.min():.4f}, {X_train.max():.4f}] (uV, no normalization)")

    return X_train, y_train


def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd


def compute_class_laterality(X, y):
    """计算每个类别的平均侧化指数"""
    laterality = torch.zeros(NUM_CLASSES)
    for cls in range(NUM_CLASSES):
        cls_data = X[y == cls]
        lat_values = []
        for i in range(len(cls_data)):
            f, psd_c3 = signal.welch(cls_data[i, C3_IDX], fs=FS, nperseg=256)
            f, psd_c4 = signal.welch(cls_data[i, C4_IDX], fs=FS, nperseg=256)
            alpha_mask = (f >= 8) & (f <= 13)
            c3_alpha = psd_c3[alpha_mask].mean()
            c4_alpha = psd_c4[alpha_mask].mean()
            lat = (c4_alpha - c3_alpha) / (c4_alpha + c3_alpha + 1e-10)
            lat_values.append(lat)
        laterality[cls] = np.mean(lat_values)
    return laterality


# ============================================================================
# 训练
# ============================================================================
def train(ddpm, loader, optimizer, scheduler, epochs,
          noise_w=1.0, spectral_w=1.0, erd_w=10.0, cls_w=5.0,
          save_path='checkpoints/bci2a_dttd_style/best_ddpm.pt',
          log_interval=25):
    print(f"\n开始训练 ({epochs} epochs, DTTD style: no normalization)")
    print(f"  损失权重: noise={noise_w}, spectral={spectral_w}, erd={erd_w}, cls={cls_w}")

    best_loss = float('inf')

    for ep in range(1, epochs + 1):
        ddpm.train()
        loss_sums = {'noise': 0, 'spectral': 0, 'erd': 0, 'classification': 0, 'total': 0}

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            loss, loss_dict = ddpm.loss(
                xb, yb,
                noise_weight=noise_w,
                spectral_weight=spectral_w,
                erd_weight=erd_w,
                cls_weight=cls_w
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.eps_model.parameters(), 1.0)
            optimizer.step()

            for key in loss_sums:
                if key in loss_dict:
                    loss_sums[key] += loss_dict[key]

        scheduler.step()

        n = len(loader)
        avg_losses = {k: v / n for k, v in loss_sums.items()}

        if ep % log_interval == 0 or ep == 1:
            print(f"Epoch {ep}: noise={avg_losses['noise']:.4f}, "
                  f"spec={avg_losses['spectral']:.4f}, "
                  f"erd={avg_losses['erd']:.4f}, "
                  f"cls={avg_losses['classification']:.4f}")

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save({
                'model_state_dict': ddpm.state_dict(),
                'target_psd': ddpm.target_psd.cpu(),
                'target_laterality': ddpm.target_laterality.cpu(),
                'epoch': ep,
                'best_loss': best_loss,
                'channels': C,
                'n_samples': T,
                'fs': FS,
                'num_classes': NUM_CLASSES,
                'data_scale': DATA_SCALE,
                'data_loader': 'BCI2a_dttd_style',
                # 不保存 data_mean/data_std — 没有归一化
            }, save_path)

    print(f"\n训练完成! 最佳损失: {best_loss:.4f}")
    print(f"模型保存到: {save_path}")
    return ddpm


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 60)
    print("BCI2a Class-Discriminative DDPM 训练 (DTTD Style: No Normalization)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"\n设备: {DEVICE}")

    # 加载数据 (不归一化)
    X_raw, y_train = load_data()

    # 计算目标统计量
    print("\n计算目标统计量...")
    target_psd = compute_target_psd(X_raw).to(DEVICE)
    target_laterality = compute_class_laterality(X_raw, y_train).to(DEVICE)

    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    print("\n目标侧化指数:")
    for cls in range(NUM_CLASSES):
        print(f"  {class_names[cls]}: {target_laterality[cls].item():+.4f}")

    # 创建模型
    print("\n创建模型...")
    eps_model = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(DEVICE)
    n_params_eps = sum(p.numel() for p in eps_model.parameters()) / 1e6
    print(f"  UNet参数量: {n_params_eps:.2f}M")

    # 训练分类器
    print(f"\n训练分类器 ({CLASSIFIER_PRETRAIN_EPOCHS} epochs)...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(DEVICE)
    classifier = pretrain_classifier(
        classifier,
        torch.FloatTensor(X_raw).to(DEVICE),
        torch.LongTensor(y_train).to(DEVICE),
        epochs=CLASSIFIER_PRETRAIN_EPOCHS,
        batch_size=64,
        lr=1e-3,
        device=DEVICE,
        save_path=f'{CHECKPOINT_DIR}/classifier.pt',
        verbose=True
    )

    # 创建DDPM
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=FS
    ).to(DEVICE)

    # 数据加载器
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_raw), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # 优化器和调度器
    optimizer = torch.optim.AdamW(eps_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # 训练
    ddpm = train(
        ddpm, loader, optimizer, scheduler,
        epochs=EPOCHS,
        noise_w=NOISE_WEIGHT,
        spectral_w=SPECTRAL_WEIGHT,
        erd_w=ERD_WEIGHT,
        cls_w=CLS_WEIGHT,
        save_path=f'{CHECKPOINT_DIR}/best_ddpm.pt',
        log_interval=25
    )

    # 测试生成
    print("\n测试生成...")
    ddpm.eval()
    with torch.no_grad():
        y_test = torch.arange(NUM_CLASSES, device=DEVICE, dtype=torch.long)
        samples = ddpm.sample_ddim(NUM_CLASSES, y_test, steps=50, guidance_scale=GUIDANCE_SCALE)
        print(f"  生成样本: shape={tuple(samples.shape)}, range=[{samples.min():.4f}, {samples.max():.4f}]")

    print("\n" + "=" * 60)
    print("BCI2a DTTD-style 训练完成!")
    print(f"模型保存在: {CHECKPOINT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
