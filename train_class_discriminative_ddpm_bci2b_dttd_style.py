#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCI2b: DTTD-style training — NO normalization, raw data

与原始 train_ddpm_once_bci2b.py 的关键区别:
1. 不做逐通道归一化 — 直接使用原始数据 (与DTTD一致)
2. 数据缩放: 原始数据 * 1e6 (微伏级别, 与DTTD一致)
3. 保存到 checkpoints/bci2b_dttd_style/ (不覆盖原始模型)
4. checkpoint 中不保存 data_mean/data_std (因为没有归一化)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "core/models/ddpm")
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    ERDLateralityLoss,
)

sys.path.insert(0, "utils")
from data_loader_bci2b import load_bci2b_data

# ============================================================================
# 配置
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DTTD-style: 数据缩放因子
DATA_SCALE = 1e6


def pick_c3_c4_indices(n_channels):
    if n_channels == 3:
        return 0, 2
    if n_channels >= 12:
        return 7, 11
    return 0, max(0, n_channels - 1)


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
    parser = argparse.ArgumentParser(description="BCI2b DTTD-style training (no normalization)")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2b")
    parser.add_argument("--out_dir", type=str, default="checkpoints/bci2b_dttd_style")
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--erd_weight", type=float, default=0.5)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--noise_weight", type=float, default=1.0)
    parser.add_argument("--spectral_weight", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("BCI2b DTTD-style training (NO normalization, raw data * 1e6)")
    print("=" * 70)
    print(f"设备: {device}\n")

    # 加载原始数据 (不归一化)
    X_raw, y, subjects, sessions, subj_map = load_bci2b_data(args.data_root, standardize=False)

    # 使用 session==0 (T) 训练
    train_mask = sessions == 0
    if not train_mask.any():
        raise RuntimeError("BCI2b 数据中不存在 session==0 (T) 的样本")
    X_train_raw = X_raw[train_mask]
    y_train = y[train_mask]

    # DTTD风格: 缩放到微伏级别，不做归一化
    X_train = (X_train_raw * DATA_SCALE).astype(np.float32)
    print(f"[BCI2b] DTTD style: 数据缩放 * {DATA_SCALE}, 不做归一化")
    print(f"[BCI2b] 缩放后范围: [{X_train.min():.4f}, {X_train.max():.4f}] (uV)")

    num_classes = int(len(np.unique(y_train)))
    C = int(X_train.shape[1])
    T = int(X_train.shape[2])

    print(f"[BCI2b] 训练集: X={X_train.shape}, y={y_train.shape}, 类别分布={np.bincount(y_train)}")

    c3_idx, c4_idx = pick_c3_c4_indices(C)
    print(f"[BCI2b] ERD通道: C3_IDX={c3_idx}, C4_IDX={c4_idx}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 训练分类器
    print(f"\n训练分类器 (50 epochs)...")
    clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(device)
    from class_discriminative import pretrain_classifier
    clf = pretrain_classifier(
        clf,
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
        save_path=os.path.join(args.out_dir, "pretrained_classifier.pt"),
        verbose=True
    )

    # 计算target
    print("\n计算 target_psd / target_laterality ...")
    tpsd, tlat = compute_targets(X_train, y_train, fs=args.fs, c3_idx=c3_idx, c4_idx=c4_idx)
    print(f"  target_laterality: {tlat.tolist()}")

    # 初始化DDPM
    print("\n初始化 DDPM 模型 ...")
    eps = MultiScaleCondUNet(channels=C, num_classes=num_classes).to(device)
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps,
        classifier=clf,
        target_psd=tpsd.to(device),
        target_laterality=tlat.to(device),
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=args.fs,
    ).to(device)

    ddpm.erd_loss_fn = ERDLateralityLoss(
        fs=args.fs, n_fft=T, c3_idx=c3_idx, c4_idx=c4_idx,
    ).to(device)

    # 训练
    print(f"\n开始训练 DDPM ({args.epochs} epochs) ...")
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    opt = torch.optim.AdamW(ddpm.eps_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                erd_weight=float(args.erd_weight),
                cls_weight=float(args.cls_weight),
                noise_weight=float(args.noise_weight),
                spectral_weight=float(args.spectral_weight),
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
        "dataset": "BCI2b_dttd_style",
        "epoch": int(args.epochs),
        "model_state_dict": ddpm.state_dict(),
        "eps_model_state_dict": ddpm.eps_model.state_dict(),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "target_psd": tpsd,
        "target_laterality": tlat,
        "channels": C,
        "n_samples": T,
        "num_classes": num_classes,
        "timesteps": 1000,
        "fs": int(args.fs),
        "erd_channels": {"c3_idx": int(c3_idx), "c4_idx": int(c4_idx)},
        "data_scale": DATA_SCALE,
        "data_loader": "BCI2b_dttd_style",
        # 不保存 data_mean/data_std — 没有归一化
        "subject_mapping": subj_map,
    }
    torch.save(ckpt, ckpt_path)
    print(f"\n已保存: {ckpt_path}")

    # 测试生成
    print("\n测试 DDIM 生成 ...")
    ddpm.eval()
    with torch.no_grad():
        y_test = torch.arange(0, num_classes, device=device, dtype=torch.long)
        samples = ddpm.sample_ddim(int(num_classes), y_test, steps=50, guidance_scale=3.0, device=str(device))
        print(f"成功生成 {len(samples)} 个样本, shape={tuple(samples.shape)}")

    print("\n" + "=" * 70)
    print("BCI2b DTTD-style 训练完成!")
    print(f"模型保存在: {args.out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
