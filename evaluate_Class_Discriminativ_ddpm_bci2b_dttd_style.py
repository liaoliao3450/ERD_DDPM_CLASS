#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCI2b: DTTD-style evaluation (NO normalization, raw data * 1e6 uV)

与原始 evaluate_Class_Discriminativ_ddpm_bci2b.py 的关键区别:
1. 不做逐通道归一化 — 数据缩放到微伏级别 (与DTTD一致)
2. DTTD-style EEGNet 分类器 (单层 separable conv)
3. DTTD 训练超参 (lr=1e-3, wd=1e-4, epochs=100, batch=32, ReduceLROnPlateau)
4. 1x 增广比例 (与真实样本等量)
5. 生成数据与真实数据在同一空间 (uV)，无需反归一化

配合训练脚本: train_class_discriminative_ddpm_bci2b_dttd_style.py
配合基线脚本: train_baselines_dttd_style.py --dataset bci2b
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

sys.path.insert(0, "core/models/ddpm")
from class_discriminative import (
    EEGClassifier,
    pretrain_classifier,
    MultiScaleCondUNet,
    ClassDiscriminativeDDPM,
)

sys.path.insert(0, "core/models/baselines")
from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 3
N_SAMPLES = 1000
FS = 250
NUM_CLASSES = 2

# DTTD-style: 数据缩放因子 (与训练一致)
DATA_SCALE = 1e6

# DTTD-style 超参数
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3
CLASSIFIER_WD = 1e-4
AUG_RATIO = 1.0


# ============================================================================
# DTTD-style EEGNet Classifier (3-channel BCI2b)
# ============================================================================

class DTTD_EEGNetClassifier(nn.Module):
    """EEGNet classifier matching DTTD project architecture (BCI2b: 3 channels)."""
    def __init__(self, num_channels=CHANNELS, num_classes=NUM_CLASSES, time_steps=N_SAMPLES):
        super().__init__()
        F1, D, F2 = 8, 2, 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(F2 * (time_steps // 32), num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.classifier(x.flatten(1))


# ============================================================================
# Data loading — DTTD style: no normalization, scale to uV
# ============================================================================

def load_bci2b_data(data_root="data/processed/BCI2b"):
    """Load BCI2b data — DTTD style: scale to uV, no normalization."""
    from data_loader_bci2b import load_bci2b_data as _load
    X, y, subjects, sessions, subj_map = _load(data_root, standardize=False)
    X = (X * DATA_SCALE).astype(np.float32)
    y = y.astype(np.int64)
    print(f"BCI2b (DTTD style): {X.shape}, range=[{X.min():.4f}, {X.max():.4f}] uV, "
          f"classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}")
    return X, y, subjects, sessions


# ============================================================================
# DTTD-style Classifier training
# ============================================================================

def train_and_eval_classifier(X_train, y_train, X_test, y_test):
    """Train DTTD-style classifier on raw uV data."""
    clf = DTTD_EEGNetClassifier().to(DEVICE)

    optimizer = torch.optim.Adam(clf.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = DataLoader(train_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, drop_last=True)

    clf.train()
    for ep in range(1, CLASSIFIER_EPOCHS + 1):
        total_loss = 0.0
        for data, labels in loader:
            optimizer.zero_grad()
            logits = clf(data.to(DEVICE))
            loss = criterion(logits, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(loader))

    clf.eval()
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False)
    preds, labels_all = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            preds.extend(torch.argmax(clf(data.to(DEVICE)), dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds)
    kappa = cohen_kappa_score(labels_all, preds)
    del clf
    torch.cuda.empty_cache()
    return acc, kappa


# ============================================================================
# Augmentation methods (operate in uV space)
# ============================================================================

def gaussian_noise_augment(X_train, y_train, n_per_class):
    gen_X, gen_y = [], []
    for c in range(NUM_CLASSES):
        class_data = X_train[y_train == c]
        class_std = class_data.std(axis=0)
        for _ in range(n_per_class):
            base = class_data[np.random.randint(len(class_data))]
            noise = np.random.randn(*base.shape) * class_std * 0.1
            gen_X.append(base + noise)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


def smote_augment(X_train, y_train, n_per_class):
    gen_X, gen_y = [], []
    for c in range(NUM_CLASSES):
        class_data = X_train[y_train == c]
        for _ in range(n_per_class):
            idx = np.random.randint(len(class_data))
            sample = class_data[idx]
            k = min(5, len(class_data) - 1)
            if k > 0:
                neighbor_idx = np.random.choice(
                    [i for i in range(len(class_data)) if i != idx], k, replace=False)
                neighbor = class_data[neighbor_idx[0]]
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
            else:
                synthetic = sample
            gen_X.append(synthetic)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


# ============================================================================
# Load models (DTTD-style checkpoints)
# ============================================================================

def load_ddpm_bci2b(checkpoint_path):
    """Load DTTD-style DDPM for BCI2b (trained on uV data, no normalization)."""
    if not os.path.exists(checkpoint_path):
        print(f"  DDPM checkpoint not found: {checkpoint_path}")
        return None
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    C = int(ckpt.get("channels", CHANNELS))
    T = int(ckpt.get("n_samples", N_SAMPLES))
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    fs = int(ckpt.get("fs", FS))

    eps_model = MultiScaleCondUNet(channels=C, num_classes=num_classes).to(DEVICE)
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=num_classes).to(DEVICE)

    if isinstance(ckpt, dict) and "target_psd" in ckpt and "target_laterality" in ckpt:
        target_psd = ckpt["target_psd"].to(DEVICE)
        target_lat = ckpt["target_laterality"].to(DEVICE)
    else:
        target_psd = torch.zeros(T // 2 + 1, device=DEVICE)
        target_lat = torch.zeros(num_classes, device=DEVICE)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_lat,
        n_timesteps=1000, channels=C, n_samples=T, fs=fs,
    ).to(DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        try:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    print(f"  Loaded DTTD-style DDPM from {checkpoint_path}")
    return ddpm


def load_baseline_bci2b(model_name):
    """Load DTTD-style baseline model for BCI2b."""
    ckpt_dir = os.path.join("checkpoints", "baselines_dttd_style")
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_bci2b.pt')
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint for {model_name}: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if model_name == 'cvae':
        model = CVAE(channels=CHANNELS, latent_dim=64, out_length=N_SAMPLES, num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(DEVICE).eval()
    elif model_name == 'wavegan':
        model = WaveGAN(channels=CHANNELS, out_length=N_SAMPLES, num_classes=NUM_CLASSES)
        model.generator.load_state_dict(ckpt['generator'])
        model.discriminator.load_state_dict(ckpt['discriminator'])
        return model.to(DEVICE).eval()
    elif model_name == 'cond_ddpm':
        model = CondDDPM(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(DEVICE).eval()
    elif model_name == 'braindiff':
        model = BrainDiff(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(DEVICE).eval()
    elif model_name == 'eegdiff':
        model = EEGDiff(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(DEVICE).eval()
    elif model_name == 'diffeegbooth':
        model = DiffEEGBooth(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES, fs=FS)
        sd = ckpt['model_state_dict']
        if 'target_laterality' in sd and sd['target_laterality'].shape[0] != NUM_CLASSES:
            sd['target_laterality'] = torch.zeros(NUM_CLASSES)
        model.load_state_dict(sd, strict=True)
        return model.to(DEVICE).eval()
    return None


def generate_baseline_samples(model, model_name, n_per_class):
    """Generate samples from DTTD-style baseline models (output in uV space)."""
    gen_X, gen_y = [], []
    GEN_BATCH = 16 if model_name in ('eegdiff', 'diffeegbooth') else n_per_class
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = n_per_class
            while remaining > 0:
                batch = min(GEN_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)
                if model_name == 'wavegan':
                    z = torch.randn(batch, model.z_dim, device=DEVICE)
                    samples = model.generator(z, y)
                elif model_name == 'cvae':
                    samples = model.generate(batch, y, DEVICE)
                elif model_name in ('cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth'):
                    samples = model.sample_ddim(batch, y, steps=50, device=DEVICE)
                else:
                    continue
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()

    gen_X = np.concatenate(gen_X)
    gen_y = np.array(gen_y)
    return gen_X, gen_y


def generate_ddpm_samples(ddpm, n_per_class, guidance_scale):
    """Generate samples from DTTD-style DDPM (output in uV space)."""
    gen_X, gen_y = [], []
    DDPM_BATCH = 16
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = n_per_class
            while remaining > 0:
                batch = min(DDPM_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample_ddim(batch, y, steps=50, guidance_scale=guidance_scale, device=str(DEVICE))
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()

    gen_X = np.concatenate(gen_X)
    gen_y = np.array(gen_y)
    return gen_X, gen_y


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y):
    """All inputs in uV space (same as training)."""
    X_aug = np.concatenate([X_train, gen_X])
    y_aug = np.concatenate([y_train, gen_y])
    return train_and_eval_classifier(X_aug, y_aug, X_test, y_test)


CACHE_DIR = "outputs/results/cache_bci2b_dttd_style"
os.makedirs(CACHE_DIR, exist_ok=True)

METHOD_NAMES = ['baseline', 'gaussian_noise', 'smote', 'cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth', 'ddpm']


def save_cache(scenario, idx, all_results):
    cache_path = os.path.join(CACHE_DIR, f"{scenario}_cache.json")
    data = {"completed_idx": idx, "results": {}}
    for name in METHOD_NAMES:
        if name in all_results and all_results[name]:
            data["results"][name] = [(float(v[0]), float(v[1])) for v in all_results[name]]
    with open(cache_path, "w") as f:
        json.dump(data, f)


def load_cache(scenario):
    cache_path = os.path.join(CACHE_DIR, f"{scenario}_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        all_results = {name: [] for name in METHOD_NAMES}
        for name, vals in data.get("results", {}).items():
            if name in all_results:
                converted = []
                for v in vals:
                    if isinstance(v, (int, float)):
                        converted.append((float(v), 0.0))
                    elif isinstance(v, (list, tuple)):
                        converted.append((float(v[0]), float(v[1])))
                    else:
                        converted.append(v)
                all_results[name] = converted
        start_idx = data.get("completed_idx", -1) + 1
        print(f"  [Resume] {scenario}: starting from subject {start_idx + 1} (cached)")
        return all_results, start_idx
    return {name: [] for name in METHOD_NAMES}, 0


# ============================================================================
# Three scenarios
# ============================================================================

def run_within_subject(X, y, subjects, ddpm, baseline_models, guidance_scale):
    print("\n" + "=" * 70)
    print("1. BCI2b Within-Subject Test — DTTD Style (uV space)")
    print("=" * 70)

    all_results, start_idx = load_cache("within_subject")
    unique_subjects = np.unique(subjects)

    for idx, subj_id in enumerate(unique_subjects):
        if idx < start_idx:
            continue
        print(f"\nSubject {idx + 1}/{len(unique_subjects)} (ID={subj_id}):")
        mask = subjects == subj_id
        X_subj, y_subj = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj)

        samples_per_class = len(X_train) // NUM_CLASSES
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES} (ratio=1.0)")

        acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append((acc, kappa))
        print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['gaussian_noise'].append((acc, kappa))
        print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['smote'].append((acc, kappa))
        print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        for name, model in baseline_models.items():
            gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results[name].append((acc, kappa))
            print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        if ddpm is not None:
            gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results['ddpm'].append((acc, kappa))
            print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        save_cache("within_subject", idx, all_results)

    summary = {}
    for name, vals in all_results.items():
        if vals:
            accs = [v[0] for v in vals]
            kappas = [v[1] for v in vals]
            summary[name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                             "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
                             "per_subject": [(float(v[0]), float(v[1])) for v in vals]}
    return summary


def run_cross_session(X, y, subjects, sessions, ddpm, baseline_models, guidance_scale):
    print("\n" + "=" * 70)
    print("2. BCI2b Cross-Session Test (T->E) — DTTD Style (uV space)")
    print("=" * 70)

    all_results, start_idx = load_cache("cross_session")
    unique_subjects = np.unique(subjects)

    for idx, subj_id in enumerate(unique_subjects):
        if idx < start_idx:
            continue
        print(f"\nSubject {idx + 1}/{len(unique_subjects)} (ID={subj_id}):")
        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)
        if not train_mask.any() or not test_mask.any():
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        samples_per_class = len(X_train) // NUM_CLASSES

        acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append((acc, kappa))
        print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['gaussian_noise'].append((acc, kappa))
        print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['smote'].append((acc, kappa))
        print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        for name, model in baseline_models.items():
            gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results[name].append((acc, kappa))
            print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        if ddpm is not None:
            gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results['ddpm'].append((acc, kappa))
            print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        save_cache("cross_session", idx, all_results)

    summary = {}
    for name, vals in all_results.items():
        if vals:
            accs = [v[0] for v in vals]
            kappas = [v[1] for v in vals]
            summary[name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                             "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
                             "per_subject": [(float(v[0]), float(v[1])) for v in vals]}
    return summary


def run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, guidance_scale):
    print("\n" + "=" * 70)
    print("3. BCI2b Cross-Subject Test (LOSO, T-only) — DTTD Style (uV space)")
    print("=" * 70)

    all_results, start_idx = load_cache("cross_subject")
    unique_subjects = np.unique(subjects)

    t_mask = sessions == 0
    if not t_mask.any():
        return {}
    X_T, y_T, subj_T = X[t_mask], y[t_mask], subjects[t_mask]

    for idx, test_subj in enumerate(unique_subjects):
        if idx < start_idx:
            continue
        print(f"\nTest subject {idx + 1}/{len(unique_subjects)} (ID={test_subj}):")
        train_mask = subj_T != test_subj
        test_mask = subj_T == test_subj
        if not train_mask.any() or not test_mask.any():
            continue

        X_train, y_train = X_T[train_mask], y_T[train_mask]
        X_test, y_test = X_T[test_mask], y_T[test_mask]
        samples_per_class = len(X_train) // NUM_CLASSES

        acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        all_results['baseline'].append((acc, kappa))
        print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['gaussian_noise'].append((acc, kappa))
        print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
        acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
        all_results['smote'].append((acc, kappa))
        print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        for name, model in baseline_models.items():
            gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results[name].append((acc, kappa))
            print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        if ddpm is not None:
            gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            all_results['ddpm'].append((acc, kappa))
            print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")

        save_cache("cross_subject", idx, all_results)

    summary = {}
    for name, vals in all_results.items():
        if vals:
            accs = [v[0] for v in vals]
            kappas = [v[1] for v in vals]
            summary[name] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                             "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
                             "per_subject": [(float(v[0]), float(v[1])) for v in vals]}
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="BCI2b DTTD-Style Evaluation (uV space, no normalization)")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2b")
    parser.add_argument("--ddpm_ckpt", type=str, default="checkpoints/bci2b_dttd_style/best_ddpm.pt")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    args = parser.parse_args()

    print("=" * 70)
    print("BCI2b DTTD-Style Evaluation (NO normalization, raw data * 1e6 uV)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Classifier: DTTD_EEGNet (lr={CLASSIFIER_LR}, wd={CLASSIFIER_WD}, epochs={CLASSIFIER_EPOCHS}, batch={CLASSIFIER_BATCH_SIZE})")
    print(f"Augmentation ratio: {AUG_RATIO}x")

    # Load data (uV space, no normalization)
    X, y, subjects, sessions = load_bci2b_data(args.data_root)

    # Load DTTD-style DDPM
    ddpm = load_ddpm_bci2b(args.ddpm_ckpt)

    # Load DTTD-style baseline models
    baseline_models = {}
    for name in ['cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth']:
        model = load_baseline_bci2b(name)
        if model is not None:
            baseline_models[name] = model
            print(f"  Loaded {name} for BCI2b (DTTD style)")

    # Run three scenarios
    results = {}
    results['within_subject'] = run_within_subject(X, y, subjects, ddpm, baseline_models, args.guidance_scale)
    results['cross_session'] = run_cross_session(X, y, subjects, sessions, ddpm, baseline_models, args.guidance_scale)
    results['cross_subject'] = run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, args.guidance_scale)

    # Print summary
    print("\n" + "=" * 70)
    print("BCI2b DTTD-Style Results Summary")
    print("=" * 70)
    for scenario, methods in results.items():
        print(f"\n{scenario}:")
        for method, vals in methods.items():
            print(f"  {method:<20}: Acc {vals['acc_mean']*100:.2f}% +/- {vals['acc_std']*100:.2f}%, Kappa {vals['kappa_mean']:.4f} +/- {vals['kappa_std']:.4f}")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/all_methods_bci2b_kappa_dttd_style.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": "BCI2b", "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
