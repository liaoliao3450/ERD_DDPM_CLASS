#!/usr/bin/env python3
"""
BCI2b Gaussian Noise 数据增强三场景评估（单独版本，不覆盖 BCI2a）

场景：
1) Within-Subject
2) Cross-Session (T->E)
3) Cross-Subject (LOSO, T-only)

说明：
- 数据来自 `data/processed/BCI2b`，通过 `utils/data_loader_bci2b.load_bci2b_data` 加载
- 分类任务为二分类（左 / 右），分类器使用 EEGClassifier(num_classes=2)
- 结果保存到 `outputs/results/gaussian_noise_bci2b_all_scenarios.json`
"""

import os
import sys
import json
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, "core/models/ddpm")
from class_discriminative import EEGClassifier, pretrain_classifier  # type: ignore

sys.path.insert(0, "utils")
from data_loader_bci2b import load_bci2b_data as load_bci2b_data_util  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bci2b_data(data_root: str = "data/processed/BCI2b") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """简单封装，保持接口统一。"""
    return load_bci2b_data_util(data_root)


def gaussian_noise_augmentation_bci2b(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples_per_class: int,
    noise_level: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    针对 BCI2b 的高斯噪声数据增强（支持任意通道数 / 时间长度 / 类别数）。

    Args:
        X_train: [N, C, T]
        y_train: [N]
        n_samples_per_class: 每类生成样本数
        noise_level: 噪声水平（相对于该类数据逐点标准差）
    """
    gen_X, gen_y = [], []
    classes = np.unique(y_train)

    for c in classes:
        class_data = X_train[y_train == c]
        if len(class_data) == 0:
            continue

        class_std = class_data.std(axis=0)  # [C, T]

        for _ in range(n_samples_per_class):
            base = class_data[np.random.randint(len(class_data))]
            noise = np.random.randn(*class_std.shape) * class_std * noise_level
            sample = base + noise
            gen_X.append(sample)
            gen_y.append(int(c))

    if not gen_X:
        return np.empty((0,) + X_train.shape[1:], dtype=X_train.dtype), np.empty((0,), dtype=y_train.dtype)

    return np.asarray(gen_X, dtype=np.float32), np.asarray(gen_y, dtype=np.int64)


def within_subject_test_bci2b(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> Dict:
    """BCI2b 被试内测试 + Gaussian Noise 增强。"""
    print("\n" + "=" * 70)
    print("1. BCI2b Within-Subject 测试（Gaussian Noise 增强）")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    num_classes = int(len(np.unique(y)))

    for idx, subj_id in enumerate(unique_subjects):
        print(f"\n被试 {idx + 1}/{n_subjects} (内部ID={subj_id}):")

        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
        )

        samples_per_class = int(len(X_train) // max(1, num_classes))
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = gaussian_noise_augmentation_bci2b(X_train, y_train, samples_per_class)
        X_train_aug = np.concatenate([X_train, gen_X], axis=0) if len(gen_X) > 0 else X_train
        y_train_aug = np.concatenate([y_train, gen_y], axis=0) if len(gen_y) > 0 else y_train

        clf = EEGClassifier(channels=X.shape[1], n_samples=X.shape[2], num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf,
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train_aug),
            epochs=100,
            batch_size=32,
            lr=1e-3,
            device=DEVICE,
            verbose=False,
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc * 100:.2f}%")

    mean_acc = float(np.mean(results)) if results else 0.0
    std_acc = float(np.std(results)) if results else 0.0

    print(f"\n平均: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    return {
        "mean": mean_acc,
        "std": std_acc,
        "per_subject": [float(a) for a in results],
    }


def cross_session_test_bci2b(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> Dict:
    """BCI2b 跨会话测试（T->E）+ Gaussian Noise 增强。"""
    print("\n" + "=" * 70)
    print("2. BCI2b Cross-Session 测试（T->E，Gaussian Noise 增强）")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    num_classes = int(len(np.unique(y)))

    for idx, subj_id in enumerate(unique_subjects):
        print(f"\n被试 {idx + 1}/{len(unique_subjects)} (内部ID={subj_id}):")

        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)

        if not train_mask.any() or not test_mask.any():
            print("  警告: 该被试缺少 T 或 E 会话，跳过。")
            continue

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        samples_per_class = int(len(X_train) // max(1, num_classes))
        print(f"  训练(T): {len(X_train)}, 测试(E): {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = gaussian_noise_augmentation_bci2b(X_train, y_train, samples_per_class)
        X_train_aug = np.concatenate([X_train, gen_X], axis=0) if len(gen_X) > 0 else X_train
        y_train_aug = np.concatenate([y_train, gen_y], axis=0) if len(gen_y) > 0 else y_train

        clf = EEGClassifier(channels=X.shape[1], n_samples=X.shape[2], num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf,
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train_aug),
            epochs=100,
            batch_size=32,
            lr=1e-3,
            device=DEVICE,
            verbose=False,
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc * 100:.2f}%")

    if not results:
        print("没有任何被试满足跨会话条件，结果为 0。")
        return {"mean": 0.0, "std": 0.0, "per_subject": []}

    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results))

    print(f"\n平均: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    return {
        "mean": mean_acc,
        "std": std_acc,
        "per_subject": [float(a) for a in results],
    }


def cross_subject_test_bci2b(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> Dict:
    """BCI2b 跨被试测试（LOSO，T-only）+ Gaussian Noise 增强。"""
    print("\n" + "=" * 70)
    print("3. BCI2b Cross-Subject 测试（LOSO, T-only，Gaussian Noise 增强）")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    num_classes = int(len(np.unique(y)))

    t_mask = sessions == 0
    if not t_mask.any():
        print("数据中不存在 session==0 (T) 的试验，无法进行跨被试测试。")
        return {"mean": 0.0, "std": 0.0, "per_subject": []}

    X_T = X[t_mask]
    y_T = y[t_mask]
    subj_T = subjects[t_mask]

    for idx, test_subj in enumerate(unique_subjects):
        print(f"\n测试被试 {idx + 1}/{len(unique_subjects)} (内部ID={test_subj}):")

        train_mask = subj_T != test_subj
        test_mask = subj_T == test_subj

        if not train_mask.any() or not test_mask.any():
            print("  警告: 该被试在 T 会话中样本不足，跳过。")
            continue

        X_train = X_T[train_mask]
        y_train = y_T[train_mask]
        X_test = X_T[test_mask]
        y_test = y_T[test_mask]

        samples_per_class = int(len(X_train) // max(1, num_classes))
        print(f"  训练: {len(X_train)} (其余被试 T), 测试: {len(X_test)} (当前被试 T), 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = gaussian_noise_augmentation_bci2b(X_train, y_train, samples_per_class)
        X_train_aug = np.concatenate([X_train, gen_X], axis=0) if len(gen_X) > 0 else X_train
        y_train_aug = np.concatenate([y_train, gen_y], axis=0) if len(gen_y) > 0 else y_train

        clf = EEGClassifier(channels=X.shape[1], n_samples=X.shape[2], num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf,
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train_aug),
            epochs=100,
            batch_size=32,
            lr=1e-3,
            device=DEVICE,
            verbose=False,
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc * 100:.2f}%")

    if not results:
        print("没有任何被试满足跨被试条件，结果为 0。")
        return {"mean": 0.0, "std": 0.0, "per_subject": []}

    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results))

    print(f"\n平均: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

    return {
        "mean": mean_acc,
        "std": std_acc,
        "per_subject": [float(a) for a in results],
    }


def main() -> None:
    print("=" * 70)
    print("BCI2b Gaussian Noise 方法评估")
    print("=" * 70)
    print(f"设备: {DEVICE}\n")

    X, y, subjects, sessions, subj_map = load_bci2b_data()

    print("\n被试映射表（原始ID -> 内部索引）:")
    for k, v in sorted(subj_map.items(), key=lambda kv: kv[1]):
        print(f"  {k} -> {v}")

    results: Dict[str, Dict] = {}
    results["within_subject"] = within_subject_test_bci2b(X, y, subjects)
    results["cross_session"] = cross_session_test_bci2b(X, y, subjects, sessions)
    results["cross_subject"] = cross_subject_test_bci2b(X, y, subjects, sessions)

    print("\n" + "=" * 70)
    print("BCI2b Gaussian Noise 最终结果汇总")
    print("=" * 70)

    print(f"\n{'场景':<20} {'平均准确率':<15} {'标准差':<10}")
    print("-" * 50)
    print(f"{'Within-Subject':<20} {results['within_subject']['mean'] * 100:>6.2f}%        {results['within_subject']['std'] * 100:>6.2f}%")
    print(f"{'Cross-Session':<20} {results['cross_session']['mean'] * 100:>6.2f}%        {results['cross_session']['std'] * 100:>6.2f}%")
    print(f"{'Cross-Subject':<20} {results['cross_subject']['mean'] * 100:>6.2f}%        {results['cross_subject']['std'] * 100:>6.2f}%")

    os.makedirs("outputs/results", exist_ok=True)
    final_results = {
        "dataset": "BCI2b",
        "method": "Gaussian Noise",
        "within_subject": results["within_subject"],
        "cross_session": results["cross_session"],
        "cross_subject": results["cross_subject"],
    }

    save_path = "outputs/results/gaussian_noise_bci2b_all_scenarios.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {save_path}")
    print("\n" + "=" * 70)
    print("BCI2b Gaussian Noise 评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()



