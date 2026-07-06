#!/usr/bin/env python3
"""
BCI2b 预训练分类器（单独版本，不覆盖 BCI2a）

输出：
- checkpoints/bci2b/pretrained_classifier.pt   （统一格式：包含 model_state_dict）

说明：
- 使用 BCI2b 的 session==0 (T) 数据作为训练（与后续 DDPM 训练对齐）
- 分类器结构复用 `core/models/ddpm/class_discriminative.py::EEGClassifier`
"""

import os
import sys
import json
import argparse
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, "core/models/ddpm")
from class_discriminative import EEGClassifier, pretrain_classifier  # type: ignore

sys.path.insert(0, "utils")
from data_loader_bci2b import load_bci2b_data  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI2b 预训练分类器（EEGNet）")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2b", help="BCI2b 处理后数据目录")
    parser.add_argument("--out_dir", type=str, default="checkpoints/bci2b", help="输出 checkpoint 目录（与 BCI2a 区分）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("BCI2b 预训练分类器（EEGNet）")
    print("=" * 70)
    print(f"设备: {device}")

    X, y, subjects, sessions, subj_map = load_bci2b_data(args.data_root, standardize=True)
    print(f"\n被试映射: {subj_map}")

    # 使用 session==0 (T) 训练，保证与 DDPM 训练集一致
    train_mask = sessions == 0
    if not train_mask.any():
        raise RuntimeError("BCI2b 数据中不存在 session==0 (T) 的样本，无法训练分类器。")

    X_train = X[train_mask]
    y_train = y[train_mask]

    num_classes = int(len(np.unique(y_train)))
    if num_classes != 2:
        print(f"⚠️  提示：检测到类别数={num_classes}，BCI2b 通常为2类；将按实际类别数构建分类器。")

    C = int(X_train.shape[1])
    T = int(X_train.shape[2])
    print(f"\n训练数据: X={X_train.shape}, y={y_train.shape}, 类别分布={np.bincount(y_train)}")

    clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(device)
    clf = pretrain_classifier(
        clf,
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=str(device),
        save_path=None,
        verbose=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "pretrained_classifier.pt")
    ckpt: Dict = {
        "dataset": "BCI2b",
        "model": "EEGClassifier(EEGNet)",
        "channels": C,
        "n_samples": T,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "subject_mapping": subj_map,
        "model_state_dict": clf.state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    print(f"\n已保存: {ckpt_path}")

    # 同时保存一份 json 配置，方便核对
    cfg_path = os.path.join(args.out_dir, "classifier_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in ckpt.items() if k != "model_state_dict"},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"配置已保存: {cfg_path}")

    print("\n" + "=" * 70)
    print("BCI2b 预训练分类器完成")
    print("=" * 70)


if __name__ == "__main__":
    main()


