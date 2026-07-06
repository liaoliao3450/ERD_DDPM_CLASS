#!/usr/bin/env python3
"""
BCI2b 训练 Class-Discriminative DDPM（单独版本，不覆盖 BCI2a）

输出（全部在 checkpoints/bci2b/ 下）：
- pretrained_classifier.pt            （如不存在可先运行 pretrain_classifier_train_bci2b.py）
- trained_ddpm.pt                     （统一 checkpoint 格式：model_state_dict + target_psd/target_laterality 等）
- ddpm_config.json

训练集约定：
- 使用 BCI2b 的 session==0 (T) 数据训练（与三场景评估脚本对齐）
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, "core/models/ddpm")
from class_discriminative import (  # type: ignore
    MultiScaleCondUNet,
    EEGClassifier,
    ClassDiscriminativeDDPM,
    ERDLateralityLoss,
)

sys.path.insert(0, "utils")
from data_loader_bci2b import load_bci2b_data  # type: ignore


def pick_c3_c4_indices(n_channels: int) -> Tuple[int, int]:
    """
    为 ERD laterality loss 选择 C3/C4 的默认索引。

    - 若通道数为 3（常见 BCI2b: C3, Cz, C4），默认 (0, 2)
    - 否则沿用 BCI2a 的默认索引 (7, 11)（前提是 n_channels 足够大）
    """
    if n_channels == 3:
        return 0, 2
    if n_channels >= 12:
        return 7, 11
    # 兜底：尽量选两端
    return 0, max(0, n_channels - 1)


def compute_targets(X: np.ndarray, y: np.ndarray, fs: int, c3_idx: int, c4_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 target_psd 与 target_laterality（按当前数据维度自适应）。"""
    T = int(X.shape[-1])
    num_classes = int(len(np.unique(y)))

    # target PSD: mean over samples+channels
    # shape: [T//2 + 1]
    fft = np.fft.rfft(X, axis=-1)
    psd = (np.abs(fft) ** 2).mean(axis=(0, 1)).astype(np.float32)

    # target laterality: per class alpha-band laterality (8-13Hz)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI2b 训练 Class-Discriminative DDPM（单独版本）")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2b", help="BCI2b 处理后数据目录")
    parser.add_argument("--out_dir", type=str, default="checkpoints/bci2b", help="输出 checkpoint 目录（与 BCI2a 区分）")
    parser.add_argument("--fs", type=int, default=250, help="采样率")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # loss weights（沿用 train_ddpm_once.py 的经验默认值）
    parser.add_argument("--erd_weight", type=float, default=0.5, help="ERD loss weight")
    parser.add_argument("--cls_weight", type=float, default=1.0, help="classification loss weight")
    parser.add_argument("--noise_weight", type=float, default=1.0, help="noise loss weight")
    parser.add_argument("--spectral_weight", type=float, default=0.5, help="spectral loss weight")
    # c3/c4 indices（可覆盖默认推断）
    parser.add_argument("--c3_idx", type=int, default=-1, help="C3 通道索引（-1 表示自动推断）")
    parser.add_argument("--c4_idx", type=int, default=-1, help="C4 通道索引（-1 表示自动推断）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("BCI2b 训练 Class-Discriminative DDPM（单独版本）")
    print("=" * 70)
    print(f"设备: {device}\n")

    X_raw, y, subjects, sessions, subj_map = load_bci2b_data(args.data_root, standardize=False)

    # 使用 session==0 (T) 训练
    train_mask = sessions == 0
    if not train_mask.any():
        raise RuntimeError("BCI2b 数据中不存在 session==0 (T) 的样本，无法训练 DDPM。")
    X_train_raw = X_raw[train_mask]
    y_train = y[train_mask]

    # 逐通道标准化 (与BCI2a/PhysioNet一致)
    data_mean = X_train_raw.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # [1, C, 1]
    data_std = X_train_raw.std(axis=(0, 2), keepdims=True).astype(np.float32)    # [1, C, 1]
    data_std = np.maximum(data_std, 1e-6)
    X_train = ((X_train_raw - data_mean) / data_std).astype(np.float32)
    clip_count = np.sum(np.abs(X_train) > 5)
    X_train = np.clip(X_train, -5.0, 5.0)
    print(f"[BCI2b] 逐通道标准化: mean={data_mean.mean():.6f}, std={data_std.mean():.6f}")
    print(f"[BCI2b] 标准化后范围: [{X_train.min():.4f}, {X_train.max():.4f}] (clipped {clip_count} values > 5)")

    num_classes = int(len(np.unique(y_train)))
    C = int(X_train.shape[1])
    T = int(X_train.shape[2])

    print(f"[BCI2b] DDPM 训练集: X={X_train.shape}, y={y_train.shape}, 类别分布={np.bincount(y_train)}")
    print(f"[BCI2b] num_classes={num_classes}, channels={C}, n_samples={T}, fs={args.fs}")

    # 选择/覆盖 C3/C4 索引
    if args.c3_idx >= 0 and args.c4_idx >= 0:
        c3_idx, c4_idx = int(args.c3_idx), int(args.c4_idx)
    else:
        c3_idx, c4_idx = pick_c3_c4_indices(C)
    print(f"[BCI2b] ERD laterality 使用通道索引: C3_IDX={c3_idx}, C4_IDX={c4_idx}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载预训练分类器（BCI2b 专用）
    clf_ckpt_path = os.path.join(args.out_dir, "pretrained_classifier.pt")
    if not os.path.exists(clf_ckpt_path):
        raise FileNotFoundError(
            f"未找到 BCI2b 预训练分类器: {clf_ckpt_path}\n"
            f"请先运行: python experiments/paper_experiments/pretrain_classifier_train_bci2b.py"
        )

    print(f"\n加载 BCI2b 预训练分类器: {clf_ckpt_path}")
    clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(device)
    clf_ckpt = torch.load(clf_ckpt_path, map_location=device)
    if isinstance(clf_ckpt, dict) and "model_state_dict" in clf_ckpt:
        clf.load_state_dict(clf_ckpt["model_state_dict"])
    else:
        clf.load_state_dict(clf_ckpt)
    clf.eval()
    print("  分类器加载成功")

    # 计算 target
    print("\n计算 target_psd / target_laterality ...")
    tpsd, tlat = compute_targets(X_train, y_train, fs=args.fs, c3_idx=c3_idx, c4_idx=c4_idx)
    print(f"  target_psd shape: {tuple(tpsd.shape)}")
    print(f"  target_laterality: {tlat.tolist()}")

    # 初始化模型
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

    # 覆盖 ERD 侧化损失的通道索引，使其适配 BCI2b 的 3 通道 (C3,Cz,C4)
    ddpm.erd_loss_fn = ERDLateralityLoss(
        fs=args.fs,
        n_fft=T,
        c3_idx=c3_idx,
        c4_idx=c4_idx,
    ).to(device)

    # 训练
    print(f"\n开始训练 DDPM ({args.epochs} epochs) ...")
    from torch.utils.data import DataLoader, TensorDataset

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
                xb,
                yb,
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

    print(f"\n训练完成！最佳Loss: {best_loss:.4f} (Epoch {best_epoch})")

    # 保存（统一 checkpoint 格式，且放在 checkpoints/bci2b/ 下）
    ckpt_path = os.path.join(args.out_dir, "trained_ddpm.pt")
    ckpt: Dict = {
        "dataset": "BCI2b",
        "epoch": int(args.epochs),
        "model_state_dict": ddpm.state_dict(),
        "eps_model_state_dict": ddpm.eps_model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
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
        "loss_weights": {
            "erd_weight": float(args.erd_weight),
            "cls_weight": float(args.cls_weight),
            "noise_weight": float(args.noise_weight),
            "spectral_weight": float(args.spectral_weight),
        },
        "seed": int(args.seed),
        "subject_mapping": subj_map,
        "data_mean": data_mean,
        "data_std": data_std,
        "data_loader": "BCI2b",
    }
    torch.save(ckpt, ckpt_path)
    print(f"\n已保存: {ckpt_path}")

    cfg_path = os.path.join(args.out_dir, "ddpm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in ckpt.items() if k not in ("model_state_dict", "eps_model_state_dict", "optimizer_state_dict", "target_psd", "target_laterality")}, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {cfg_path}")

    # 简单生成测试（确保可采样）
    print("\n测试 DDIM 生成 ...")
    ddpm.eval()
    with torch.no_grad():
        y_test = torch.arange(0, num_classes, device=device, dtype=torch.long)
        samples = ddpm.sample_ddim(int(num_classes), y_test, steps=50, guidance_scale=3.0, device=str(device))
        print(f"成功生成 {len(samples)} 个样本, shape={tuple(samples.shape)}")

    print("\n" + "=" * 70)
    print("BCI2b DDPM 训练并保存完成（未覆盖任何 BCI2a 文件）")
    print("=" * 70)


if __name__ == "__main__":
    main()


