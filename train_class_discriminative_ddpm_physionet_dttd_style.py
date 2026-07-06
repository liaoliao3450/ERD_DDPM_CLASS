#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysioNet MI4C: DTTD-style training — NO normalization, raw data

与原始 train_class_discriminative_ddpm_physionet.py 的关键区别:
1. 不做逐通道归一化 — 直接使用原始数据 (与DTTD一致)
2. 数据缩放: 原始数据 * 1e6 (微伏级别, 与DTTD一致)
3. 保存到 checkpoints/physionet_dttd_style/ (不覆盖原始模型)
4. checkpoint 中不保存 data_mean/data_std (因为没有归一化)
"""
import os
import sys
import io
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from data_loader_physionet_mi4c import load_physionet_mi4c_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core/models/ddpm"))
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    pretrain_classifier
)

# ============================================================================
# 配置
# ============================================================================
CHECKPOINT_DIR = 'checkpoints/physionet_dttd_style'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHANNELS = 64
N_SAMPLES = 640
FS = 160
NUM_CLASSES = 4

EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

NOISE_WEIGHT = 1.0
SPECTRAL_WEIGHT = 0.5
ERD_WEIGHT = 0.5
CLS_WEIGHT = 1.0

GUIDANCE_SCALE = 7.0
CLASSIFIER_PRETRAIN_EPOCHS = 200

# DTTD-style: 数据缩放因子
DATA_SCALE = 1e6


def compute_targets(X, y, fs, c3_idx, c4_idx):
    T = int(X.shape[-1])
    num_classes = int(len(np.unique(y)))

    fft = np.fft.rfft(X, axis=-1)
    psd = (np.abs(fft) ** 2).mean(axis=(0, 1)).astype(np.float32)

    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)

    lat = []
    for c in range(num_classes):
        m = y == c
        if int(m.sum()) == 0:
            lat.append(0.0)
            continue
        d = X[m]
        c3 = (np.abs(np.fft.rfft(d[:, c3_idx, :], axis=-1)) ** 2)[:, alpha_mask].mean()
        c4 = (np.abs(np.fft.rfft(d[:, c4_idx, :], axis=-1)) ** 2)[:, alpha_mask].mean()
        lat.append(float((c4 - c3) / (c4 + c3 + 1e-10)))

    return torch.tensor(psd, dtype=torch.float32), torch.tensor(lat, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="PhysioNet DTTD-style training (no normalization)")
    parser.add_argument("--data_root", type=str, default="data/processed/PhysioNetMI4C")
    parser.add_argument("--out_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("PhysioNet MI4C DTTD-style training (NO normalization, raw data * 1e6)")
    print("=" * 70)
    print(f"设备: {device}\n")

    # 加载原始数据 (不归一化)
    X_raw, y, subjects, sessions, _ = load_physionet_mi4c_data(args.data_root, standardize=False)
    X_raw = X_raw.astype(np.float32)
    y = y.astype(np.int64)
    y = y - y.min()
    mask = y < NUM_CLASSES
    X_raw, y = X_raw[mask], y[mask]

    # DTTD风格: 缩放到微伏级别，不做归一化
    X_train = (X_raw * DATA_SCALE).astype(np.float32)
    print(f"[PhysioNet] DTTD style: 数据缩放 * {DATA_SCALE}, 不做归一化")
    print(f"[PhysioNet] 缩放后范围: [{X_train.min():.4f}, {X_train.max():.4f}] (uV)")
    print(f"[PhysioNet] 训练集: X={X_train.shape}, 类别分布={[sum(y==i) for i in range(NUM_CLASSES)]}")

    C = int(X_train.shape[1])
    T = int(X_train.shape[2])

    # PhysioNet 64通道: C3/C4 近似索引
    c3_idx, c4_idx = 26, 30
    print(f"[PhysioNet] ERD通道: C3_IDX={c3_idx}, C4_IDX={c4_idx}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 训练分类器
    print(f"\n训练分类器 ({CLASSIFIER_PRETRAIN_EPOCHS} epochs)...")
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    classifier = pretrain_classifier(
        classifier,
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y).to(device),
        epochs=CLASSIFIER_PRETRAIN_EPOCHS,
        batch_size=64,
        lr=1e-3,
        device=device,
        save_path=os.path.join(args.out_dir, "pretrained_classifier.pt"),
        verbose=True
    )

    # 计算target
    print("\n计算 target_psd / target_laterality ...")
    tpsd, tlat = compute_targets(X_train, y, fs=FS, c3_idx=c3_idx, c4_idx=c4_idx)
    print(f"  target_laterality: {tlat.tolist()}")

    # 初始化DDPM
    print("\n初始化 DDPM 模型 ...")
    eps = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(device)
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps,
        classifier=classifier,
        target_psd=tpsd.to(device),
        target_laterality=tlat.to(device),
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=FS,
    ).to(device)

    # 训练
    print(f"\n开始训练 DDPM ({args.epochs} epochs) ...")
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    opt = torch.optim.AdamW(ddpm.eps_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float("inf")
    best_epoch = 0

    for e in range(args.epochs):
        ddpm.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss, _ = ddpm.loss(
                xb, yb,
                erd_weight=ERD_WEIGHT,
                cls_weight=CLS_WEIGHT,
                noise_weight=NOISE_WEIGHT,
                spectral_weight=SPECTRAL_WEIGHT,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = e + 1

        if (e + 1) % 10 == 0 or e == 0:
            print(f"  Epoch {e+1}/{args.epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (Epoch {best_epoch})")

    print(f"\n训练完成! 最佳Loss: {best_loss:.4f} (Epoch {best_epoch})")

    # 保存
    ckpt_path = os.path.join(args.out_dir, "trained_ddpm.pt")
    ckpt = {
        "dataset": "PhysioNet_dttd_style",
        "epoch": int(args.epochs),
        "model_state_dict": ddpm.state_dict(),
        "eps_model_state_dict": ddpm.eps_model.state_dict(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "target_psd": tpsd,
        "target_laterality": tlat,
        "channels": C,
        "n_samples": T,
        "num_classes": NUM_CLASSES,
        "timesteps": 1000,
        "fs": FS,
        "data_scale": DATA_SCALE,
        "data_loader": "PhysioNet_dttd_style",
        # 不保存 data_mean/data_std — 没有归一化
    }
    torch.save(ckpt, ckpt_path)
    print(f"\n已保存: {ckpt_path}")

    # 测试生成
    print("\n测试 DDIM 生成 ...")
    ddpm.eval()
    with torch.no_grad():
        y_test = torch.arange(0, NUM_CLASSES, device=device, dtype=torch.long)
        samples = ddpm.sample_ddim(int(NUM_CLASSES), y_test, steps=50, guidance_scale=GUIDANCE_SCALE)
        print(f"成功生成 {len(samples)} 个样本, shape={tuple(samples.shape)}")

    print("\n" + "=" * 70)
    print("PhysioNet DTTD-style 训练完成!")
    print(f"模型保存在: {args.out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
