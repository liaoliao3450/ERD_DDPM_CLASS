#!/usr/bin/env python3
"""
BCI2b：GAN / VAE 三场景分类评估（单独版本，不覆盖 BCI2a）

场景：
1) Within-Subject
2) Cross-Session (T->E)
3) Cross-Subject (LOSO, T-only)

依赖：
- checkpoints/bci2b/gan_bci2b.pt   （由 train_gan_vae_bci2b.py 生成）
- checkpoints/bci2b/vae_bci2b.pt   （由 train_gan_vae_bci2b.py 生成）

输出：
- outputs/results/gan_bci2b_all_scenarios.json
- outputs/results/vae_bci2b_all_scenarios.json
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROEJCT_FILE = Path(__file__).resolve()
# 当前文件在: experiments/paper_experiments/，项目根目录在其上两级再往上一层
PROJECT_ROOT = PROEJCT_FILE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 GAN / VAE 结构
import importlib.util

gan_model_path = PROJECT_ROOT / "core" / "models" / "gan" / "model.py"
spec = importlib.util.spec_from_file_location("gan_model", gan_model_path)
gan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan_module)
Gen1D = gan_module.Gen1D

vae_model_path = PROJECT_ROOT / "core" / "models" / "vae" / "vae_model.py"
spec = importlib.util.spec_from_file_location("vae_model", vae_model_path)
vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae_module)

# 分类器与数据加载
sys.path.insert(0, str(PROJECT_ROOT / "core" / "models" / "ddpm"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from class_discriminative import EEGClassifier, pretrain_classifier  # type: ignore
from data_loader_bci2b import load_bci2b_data  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gan_model_bci2b(C: int, T: int, num_classes: int) -> torch.nn.Module:
    """加载 BCI2b GAN 生成器。"""
    ckpt_path = "checkpoints/bci2b/gan_bci2b.pt"
    if not os.path.exists(ckpt_path):
        print(f"未找到 BCI2b GAN checkpoint: {ckpt_path}")
        return None

    try:
        G = Gen1D(z_dim=128, out_channels=C, out_length=T, num_classes=num_classes, cond_embed_dim=32).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(state["G"])
        G.eval()
        print(f"已加载 BCI2b GAN: {ckpt_path}")
        return G
    except Exception as e:
        print(f"加载 BCI2b GAN 失败: {e}")
        return None


def load_vae_model_bci2b(C: int, T: int, num_classes: int) -> torch.nn.Module:
    """加载 BCI2b VAE。"""
    ckpt_path = "checkpoints/bci2b/vae_bci2b.pt"
    if not os.path.exists(ckpt_path):
        print(f"未找到 BCI2b VAE checkpoint: {ckpt_path}")
        return None

    try:
        vae = vae_module.VAE1D(channels=C, length=T, latent_dim=128, cond_dim=32, num_classes=num_classes).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        vae.load_state_dict(state["model"])
        vae.eval()
        print(f"已加载 BCI2b VAE: {ckpt_path}")
        return vae
    except Exception as e:
        print(f"加载 BCI2b VAE 失败: {e}")
        return None


def generate_gan_samples_bci2b(G: torch.nn.Module, n_samples_per_class: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用 BCI2b GAN 生成样本。"""
    gen_X, gen_y = [], []
    if G is None or n_samples_per_class <= 0:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)

    with torch.no_grad():
        for c in range(num_classes):
            made = 0
            while made < n_samples_per_class:
                batch_size = min(50, n_samples_per_class - made)
                z = torch.randn(batch_size, 128, device=DEVICE)
                y_cond = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = G(z, y_cond)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)
                made += batch_size

    return np.concatenate(gen_X, axis=0), np.asarray(gen_y, dtype=np.int64)


def generate_vae_samples_bci2b(vae: torch.nn.Module, n_samples_per_class: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """使用 BCI2b VAE 生成样本。"""
    gen_X, gen_y = [], []
    if vae is None or n_samples_per_class <= 0:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)

    with torch.no_grad():
        for c in range(num_classes):
            made = 0
            while made < n_samples_per_class:
                batch_size = min(50, n_samples_per_class - made)
                z = torch.randn(batch_size, 128, device=DEVICE)
                y_cond = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = vae.decode(z, y_cond)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)
                made += batch_size

    return np.concatenate(gen_X, axis=0), np.asarray(gen_y, dtype=np.int64)


def within_subject_test_bci2b(model, model_name: str, generate_func, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, num_classes: int) -> Dict:
    """BCI2b 被试内测试（增强方法）。"""
    print("\n" + "=" * 70)
    print(f"1. BCI2b Within-Subject 测试 - {model_name}")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    C, T = X.shape[1], X.shape[2]

    for idx, subj_id in enumerate(unique_subjects):
        print(f"\n被试 {idx + 1}/{len(unique_subjects)} (内部ID={subj_id}):")

        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
        )

        samples_per_class = int(len(X_train) // max(1, num_classes))
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = generate_func(model, samples_per_class, num_classes)
        if gen_X.size == 0:
            X_train_aug, y_train_aug = X_train, y_train
        else:
            X_train_aug = np.concatenate([X_train, gen_X], axis=0)
            y_train_aug = np.concatenate([y_train, gen_y], axis=0)

        clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(DEVICE)
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

    return {"mean": mean_acc, "std": std_acc, "per_subject": [float(a) for a in results]}


def cross_session_test_bci2b(model, model_name: str, generate_func, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, num_classes: int) -> Dict:
    """BCI2b 跨会话测试（T->E，增强方法）。"""
    print("\n" + "=" * 70)
    print(f"2. BCI2b Cross-Session 测试（T->E） - {model_name}")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    C, T = X.shape[1], X.shape[2]

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

        gen_X, gen_y = generate_func(model, samples_per_class, num_classes)
        if gen_X.size == 0:
            X_train_aug, y_train_aug = X_train, y_train
        else:
            X_train_aug = np.concatenate([X_train, gen_X], axis=0)
            y_train_aug = np.concatenate([y_train, gen_y], axis=0)

        clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(DEVICE)
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

    return {"mean": mean_acc, "std": std_acc, "per_subject": [float(a) for a in results]}


def cross_subject_test_bci2b(model, model_name: str, generate_func, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, num_classes: int) -> Dict:
    """BCI2b 跨被试测试（LOSO, T-only，增强方法）。"""
    print("\n" + "=" * 70)
    print(f"3. BCI2b Cross-Subject 测试（LOSO, T-only） - {model_name}")
    print("=" * 70)

    results = []
    unique_subjects = np.unique(subjects)
    C, T = X.shape[1], X.shape[2]

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

        gen_X, gen_y = generate_func(model, samples_per_class, num_classes)
        if gen_X.size == 0:
            X_train_aug, y_train_aug = X_train, y_train
        else:
            X_train_aug = np.concatenate([X_train, gen_X], axis=0)
            y_train_aug = np.concatenate([y_train, gen_y], axis=0)

        clf = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(DEVICE)
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

    return {"mean": mean_acc, "std": std_acc, "per_subject": [float(a) for a in results]}


def evaluate_model_bci2b(model_name: str, model, generate_func, X: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, num_classes: int) -> Dict:
    """评估单个 BCI2b 模型（GAN 或 VAE）。"""
    print("\n" + "=" * 70)
    print(f"BCI2b {model_name} 方法评估")
    print("=" * 70)
    print(f"设备: {DEVICE}\n")

    if model is None:
        print(f"{model_name} 模型加载失败，跳过评估")
        return {}

    results: Dict[str, Dict] = {}
    results["within_subject"] = within_subject_test_bci2b(model, model_name, generate_func, X, y, subjects, num_classes)
    results["cross_session"] = cross_session_test_bci2b(model, model_name, generate_func, X, y, subjects, sessions, num_classes)
    results["cross_subject"] = cross_subject_test_bci2b(model, model_name, generate_func, X, y, subjects, sessions, num_classes)

    print("\n" + "=" * 70)
    print(f"BCI2b {model_name} 最终结果汇总")
    print("=" * 70)
    print(f"\n{'场景':<20} {'平均准确率':<15} {'标准差':<10}")
    print("-" * 50)
    print(f"{'Within-Subject':<20} {results['within_subject']['mean'] * 100:>6.2f}%        {results['within_subject']['std'] * 100:>6.2f}%")
    print(f"{'Cross-Session':<20} {results['cross_session']['mean'] * 100:>6.2f}%        {results['cross_session']['std'] * 100:>6.2f}%")
    print(f"{'Cross-Subject':<20} {results['cross_subject']['mean'] * 100:>6.2f}%        {results['cross_subject']['std'] * 100:>6.2f}%")

    os.makedirs("outputs/results", exist_ok=True)
    out = {
        "dataset": "BCI2b",
        "method": model_name,
        "within_subject": results["within_subject"],
        "cross_session": results["cross_session"],
        "cross_subject": results["cross_subject"],
    }
    out_path = f"outputs/results/{model_name.lower()}_bci2b_all_scenarios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {out_path}")

    return results


def main() -> None:
    print("=" * 70)
    print("BCI2b GAN / VAE 三场景评估")
    print("=" * 70)

    X, y, subjects, sessions, subj_map = load_bci2b_data()
    C, T = X.shape[1], X.shape[2]
    num_classes = int(len(np.unique(y)))
    print(f"数据: X={X.shape}, y={y.shape}, 通道数={C}, 采样点={T}, 类别数={num_classes}")

    print("\n被试映射表（原始ID -> 内部索引）:")
    for k, v in sorted(subj_map.items(), key=lambda kv: kv[1]):
        print(f"  {k} -> {v}")

    # 评估 GAN
    print("\n加载 BCI2b GAN 模型 ...")
    gan_model = load_gan_model_bci2b(C, T, num_classes)
    if gan_model is not None:
        evaluate_model_bci2b("GAN", gan_model, generate_gan_samples_bci2b, X, y, subjects, sessions, num_classes)

    # 评估 VAE
    print("\n加载 BCI2b VAE 模型 ...")
    vae_model = load_vae_model_bci2b(C, T, num_classes)
    if vae_model is not None:
        evaluate_model_bci2b("VAE", vae_model, generate_vae_samples_bci2b, X, y, subjects, sessions, num_classes)

    print("\n" + "=" * 70)
    print("BCI2b GAN / VAE 三场景评估完成")
    print("=" * 70)


if __name__ == "__main__":
    main()



