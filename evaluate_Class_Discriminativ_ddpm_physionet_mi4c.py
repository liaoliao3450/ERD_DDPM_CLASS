#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysioNet MI4C: All methods three-scenario classification evaluation

Key design: Normalization consistency
- All models (DDPM + baselines) were trained on data normalized with specific mean/std
- We load the normalization stats from the DDPM checkpoint
- Both real data and generated data are in the SAME normalized space
- No per-split re-normalization (which would break consistency with generated data)

Methods: Baseline, Gaussian Noise, SMOTE, CVAE, WaveGAN, Cond-DDPM, BrainDiff, EEGDiff, DiffEEGBooth, DDPM (Ours)
Scenarios:
1) Within-Subject
2) Cross-Session (Session 0->1)
3) Cross-Subject (LMSO 10-Fold)
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
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

sys.path.insert(0, "utils")
from data_loader_physionet_mi4c import load_physionet_mi4c_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 64
N_SAMPLES = 640
FS = 160
NUM_CLASSES = 4
CLASSIFIER_EPOCHS = 300
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3


# ============================================================================
# Data loading — NO global normalization here, we use checkpoint stats
# ============================================================================

def load_physionet_data(data_root="data/processed/PhysioNetMI4C"):
    """Load PhysioNet MI4C raw data (no normalization)."""
    X, y, subjects, sessions, _ = load_physionet_mi4c_data(data_root=data_root)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Ensure labels are 0-indexed
    y = y - y.min()
    mask = y < NUM_CLASSES
    X, y = X[mask], y[mask]
    subjects = subjects[mask]
    sessions = sessions[mask]

    print(f"PhysioNet MI4C raw: {X.shape}, range=[{X.min():.6f}, {X.max():.6f}], "
          f"classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}, sessions: {np.unique(sessions)}")
    return X, y, subjects, sessions


def normalize_with_stats(X, data_mean, data_std):
    """Normalize data using pre-computed statistics (same as training)."""
    X_norm = ((X - data_mean) / data_std).astype(np.float32)
    X_norm = np.clip(X_norm, -5.0, 5.0)  # Same clip as training
    return X_norm


# ============================================================================
# Classifier training with validation-based model selection
# ============================================================================

def train_and_eval_classifier(X_train, y_train, X_test, y_test, use_val=False):
    """Train classifier and return (accuracy, kappa).
    No validation split — use training accuracy for model selection (consistent with original evaluation).
    Data is already normalized with DDPM training stats, no extra normalization needed.
    Keeps data on CPU and moves mini-batches to GPU to avoid OOM when DDPM model is on GPU.
    """
    clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(clf.parameters(), lr=CLASSIFIER_LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS, eta_min=1e-5)

    # Keep data on CPU to save GPU memory (DDPM model already occupies ~12GB)
    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_train)

    best_acc = 0.0
    best_state = None

    for ep in range(1, CLASSIFIER_EPOCHS + 1):
        clf.train()
        indices = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), CLASSIFIER_BATCH_SIZE):
            batch_idx = indices[i:i + CLASSIFIER_BATCH_SIZE]
            xb = X_tr_t[batch_idx].to(DEVICE)
            yb = y_tr_t[batch_idx].to(DEVICE)
            logits = clf(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if ep % 10 == 0 or ep == CLASSIFIER_EPOCHS:
            clf.eval()
            with torch.no_grad():
                # Batch evaluation to avoid OOM on large datasets
                all_preds = []
                for j in range(0, len(X_tr_t), CLASSIFIER_BATCH_SIZE):
                    batch_logits = clf(X_tr_t[j:j+CLASSIFIER_BATCH_SIZE].to(DEVICE))
                    all_preds.append(batch_logits.argmax(1))
                train_pred = torch.cat(all_preds)
                y_tr_dev = y_tr_t[:len(train_pred)].to(DEVICE)
                train_acc = (train_pred == y_tr_dev).float().mean().item()
            if train_acc > best_acc:
                best_acc = train_acc
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        # Batch evaluation on test set
        all_preds = []
        X_test_t = torch.FloatTensor(X_test)
        for j in range(0, len(X_test_t), CLASSIFIER_BATCH_SIZE):
            batch_logits = clf(X_test_t[j:j+CLASSIFIER_BATCH_SIZE].to(DEVICE))
            all_preds.append(batch_logits.argmax(1).cpu())
        pred = torch.cat(all_preds).numpy()
    acc = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    del clf
    torch.cuda.empty_cache()
    return acc, kappa


# ============================================================================
# Augmentation methods (operate in normalized space)
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
# Load models
# ============================================================================

def load_ddpm_physionet(checkpoint_path):
    """Load DDPM and return (model, checkpoint_dict). Checkpoint contains data_mean/data_std."""
    if not os.path.exists(checkpoint_path):
        print(f"  DDPM checkpoint not found: {checkpoint_path}")
        return None, None, None, None
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(DEVICE)

    if isinstance(ckpt, dict) and "target_psd" in ckpt and "target_laterality" in ckpt:
        target_psd = ckpt["target_psd"].to(DEVICE)
        target_lat = ckpt["target_laterality"].to(DEVICE)
    else:
        target_psd = torch.zeros(N_SAMPLES // 2 + 1, device=DEVICE)
        target_lat = torch.zeros(NUM_CLASSES, device=DEVICE)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_lat,
        n_timesteps=1000, channels=CHANNELS, n_samples=N_SAMPLES, fs=FS,
    ).to(DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        try:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError:
            ddpm.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()

    # Extract normalization stats from checkpoint
    data_mean = None
    data_std = None
    if isinstance(ckpt, dict) and "data_mean" in ckpt and "data_std" in ckpt:
        data_mean = ckpt["data_mean"]  # numpy array [1, C, 1]
        data_std = ckpt["data_std"]    # numpy array [1, C, 1]
        print(f"  Loaded normalization stats from checkpoint: mean={data_mean.mean():.6f}, std={data_std.mean():.6f}")
    else:
        print(f"  WARNING: checkpoint has no data_mean/data_std, will compute from current data")

    print(f"  Loaded DDPM from {checkpoint_path}")
    return ddpm, ckpt, data_mean, data_std


def load_baseline_physionet(model_name):
    """Load a trained baseline model for PhysioNet."""
    ckpt_dir = os.path.join("checkpoints", "baselines")
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_physionet.pt')
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
    """Generate samples from baseline models. Output is in normalized space (clipped to [-5, 5])."""
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
                    # Use standard DDPM sampling (1000 steps) for fair comparison with original papers
                    samples = model.sample(batch, y, device=DEVICE)
                else:
                    continue
                # Clip to same range as training data (normalized + clipped)
                samples = torch.clamp(samples, -5.0, 5.0)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()
    return np.concatenate(gen_X), np.array(gen_y)


def generate_ddpm_samples(ddpm, n_per_class, guidance_scale=2.0, eta=0.0):
    """Generate samples from DDPM. Output is in normalized space (clipped to [-5, 5])."""
    gen_X, gen_y = [], []
    DDPM_BATCH = 16
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = n_per_class
            while remaining > 0:
                batch = min(DDPM_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample(batch, y, guidance_scale=guidance_scale, device=str(DEVICE))
                # Clip to same range as training data (normalized + clipped)
                samples = torch.clamp(samples, -5.0, 5.0)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()
    return np.concatenate(gen_X), np.array(gen_y)


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y, use_val=True):
    """All inputs must be in the same normalized space."""
    X_aug = np.concatenate([X_train, gen_X])
    y_aug = np.concatenate([y_train, gen_y])
    return train_and_eval_classifier(X_aug, y_aug, X_test, y_test, use_val=use_val)


CACHE_DIR = "outputs/results/cache_physionet_v3_aug02"
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
                        converted.append((float(v), 0.0))  # old format: acc only
                    elif isinstance(v, (list, tuple)):
                        converted.append((float(v[0]), float(v[1])))
                    else:
                        converted.append(v)
                all_results[name] = converted
        start_idx = data.get("completed_idx", -1) + 1
        print(f"  [Resume] {scenario}: starting from index {start_idx + 1} (cached)")
        return all_results, start_idx
    return {name: [] for name in METHOD_NAMES}, 0


# ============================================================================
# Cross-Subject scenario
# ============================================================================

def _should_run(method_name, skip_methods, all_results, idx):
    """Check if a method should be run (not skipped and not already cached for this subject)."""
    if method_name not in skip_methods:
        return True
    if len(all_results.get(method_name, [])) <= idx:
        return True
    return False


def _store_result(all_results, method_name, idx, acc, kappa):
    """Store result, handling both new and existing entries."""
    if len(all_results.get(method_name, [])) > idx:
        all_results[method_name][idx] = (acc, kappa)
    else:
        all_results[method_name].append((acc, kappa))


def run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, guidance_scale, eta=0.0, aug_ratio=0.2, run_methods=None):
    """LMSO 10-Fold Cross-Subject
    aug_ratio: ratio of generated samples to real training samples per class (e.g. 0.2 = 20%)
    """
    print("\n" + "=" * 70)
    print("3. PhysioNet MI4C Cross-Subject Test (LMSO 10-Fold)")
    print("=" * 70)

    all_results, start_idx = load_cache("cross_subject")
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    print(f"Total subjects: {n_subjects}")

    if run_methods is not None:
        skip_methods = set(METHOD_NAMES) - set(run_methods)
    else:
        skip_methods = set()

    np.random.seed(42)
    shuffled_ids = np.random.permutation(unique_subjects).tolist()
    n_folds = 10
    fold_size = n_subjects // n_folds
    remainder = n_subjects % n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled_ids[start:start + size])
        start += size

    for i, fold_subjects in enumerate(folds):
        print(f"  Fold {i+1} test subjects ({len(fold_subjects)}): {fold_subjects[:3]}...")

    for fold_idx in range(n_folds):
        if fold_idx < start_idx:
            continue
        test_subjects = folds[fold_idx]
        train_subjects = [s for s in unique_subjects if s not in test_subjects]
        print(f"\n--- Fold {fold_idx+1}/{n_folds} (test: {len(test_subjects)} subjects, train: {len(train_subjects)} subjects) ---")

        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, test_subjects)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        samples_per_class = max(10, int(len(X_train) // NUM_CLASSES * aug_ratio))
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES} (ratio={aug_ratio})")

        # Baseline (use_val=False: cross-subject, no validation split)
        if _should_run('baseline', skip_methods, all_results, fold_idx):
            acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test, use_val=False)
            _store_result(all_results, 'baseline', fold_idx, acc, kappa)
            print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Baseline: [cached]")

        # Gaussian Noise
        if _should_run('gaussian_noise', skip_methods, all_results, fold_idx):
            gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y, use_val=False)
            _store_result(all_results, 'gaussian_noise', fold_idx, acc, kappa)
            print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Gaussian Noise: [cached]")

        # SMOTE
        if _should_run('smote', skip_methods, all_results, fold_idx):
            gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y, use_val=False)
            _store_result(all_results, 'smote', fold_idx, acc, kappa)
            print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  SMOTE: [cached]")

        # Baseline models
        for name, model in baseline_models.items():
            if _should_run(name, skip_methods, all_results, fold_idx):
                gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y, use_val=False)
                _store_result(all_results, name, fold_idx, acc, kappa)
                print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  {name}: [cached]")

        # DDPM
        if ddpm is not None:
            if _should_run('ddpm', skip_methods, all_results, fold_idx):
                gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale, eta)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y, use_val=False)
                _store_result(all_results, 'ddpm', fold_idx, acc, kappa)
                print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  DDPM: [cached]")

        save_cache("cross_subject", fold_idx, all_results)

    summary = {}
    for name, vals in all_results.items():
        if vals:
            accs = [v[0] for v in vals]
            kappas = [v[1] for v in vals]
            summary[name] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "kappa_mean": float(np.mean(kappas)), "kappa_std": float(np.std(kappas)),
                "per_fold": [(float(v[0]), float(v[1])) for v in vals]
            }
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PhysioNet MI4C All Methods Evaluation")
    parser.add_argument("--data_root", type=str, default="data/processed/PhysioNetMI4C")
    parser.add_argument("--ddpm_ckpt", type=str, default="checkpoints/best_class_discriminative_physionet_mi4c.pt")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0=deterministic, 1=DDPM)")
    parser.add_argument("--aug_ratio", type=float, default=0.2, help="Augmentation ratio (0.2 = 20%% of real samples per class)")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods to re-run (others use cache). "
                             "Options: baseline,gaussian_noise,smote,cvae,wavegan,cond_ddpm,braindiff,eegdiff,diffeegbooth,ddpm")
    args = parser.parse_args()

    # Parse methods filter
    run_methods = None
    if args.methods:
        run_methods = [m.strip() for m in args.methods.split(',')]
        print(f"[Selective] Only re-running: {run_methods} (others use cache)")

    print("=" * 70)
    print("PhysioNet MI4C All Methods Three-Scenario Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load raw data (no normalization)
    X_raw, y, subjects, sessions = load_physionet_data(args.data_root)

    # Load DDPM (with normalization stats)
    ddpm, ckpt, data_mean, data_std = load_ddpm_physionet(args.ddpm_ckpt)

    # Normalize raw data using checkpoint stats (same as training)
    if data_mean is not None and data_std is not None:
        X = normalize_with_stats(X_raw, data_mean, data_std)
        print(f"  Normalized with checkpoint stats: range=[{X.min():.4f}, {X.max():.4f}]")
    else:
        # Fallback: compute stats from current data
        data_mean = X_raw.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        data_std = np.maximum(X_raw.std(axis=(0, 2), keepdims=True).astype(np.float32), 1e-6)
        X = normalize_with_stats(X_raw, data_mean, data_std)
        print(f"  Normalized with computed stats (no checkpoint): range=[{X.min():.4f}, {X.max():.4f}]")

    # Load baseline models (only those needed)
    needed_baselines = ['cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth']
    if run_methods is not None:
        needed_baselines = [n for n in needed_baselines if n in run_methods]

    baseline_models = {}
    for name in needed_baselines:
        model = load_baseline_physionet(name)
        if model is not None:
            baseline_models[name] = model
            print(f"  Loaded {name} for PhysioNet")

    # Run cross-subject only
    results = {}
    results['cross_subject'] = run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, args.guidance_scale, args.eta, args.aug_ratio, run_methods)

    # Print summary
    print("\n" + "=" * 70)
    print("PhysioNet MI4C Final Results Summary")
    print("=" * 70)
    for scenario, methods in results.items():
        print(f"\n{scenario}:")
        for method, vals in methods.items():
            print(f"  {method:<20}: Acc {vals['acc_mean']*100:.2f}% +/- {vals['acc_std']*100:.2f}%, Kappa {vals['kappa_mean']:.4f} +/- {vals['kappa_std']:.4f}")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/all_methods_physionet_kappa_v3_aug02.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
