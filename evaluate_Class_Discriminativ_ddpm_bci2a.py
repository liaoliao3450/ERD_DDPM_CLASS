#!/usr/bin/env python3
"""
BCI2a: All methods three-scenario classification evaluation

Methods: Baseline, Gaussian Noise, SMOTE, CVAE, WaveGAN, Cond-DDPM, BrainDiff, EEGDiff, DiffEEGBooth, DDPM (Ours)
Scenarios:
1) Within-Subject
2) Cross-Session (T->E)
3) Cross-Subject (LOSO, T-only)
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 22
N_SAMPLES = 1000
FS = 250
NUM_CLASSES = 4
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3


# ============================================================================
# Data loading
# ============================================================================

def load_bci2a_data(data_root="data/processed/BCI2a"):
    """Load BCI2a raw data (no normalization — we use checkpoint stats)."""
    X = np.load(os.path.join(data_root, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(data_root, "y.npy")).astype(np.int64)

    y = y - y.min()
    mask = y < NUM_CLASSES
    X, y = X[mask], y[mask]

    n_subjects = 9
    n_sessions = 2
    trials_per_session = len(X) // (n_subjects * n_sessions)

    subjects = []
    sessions = []
    for subj in range(n_subjects):
        for sess in range(n_sessions):
            subjects.extend([subj] * trials_per_session)
            sessions.extend([sess] * trials_per_session)

    subjects = np.array(subjects[:len(X)])
    sessions = np.array(sessions[:len(X)])

    print(f"BCI2a raw: {X.shape}, range=[{X.min():.6f}, {X.max():.6f}], classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}")
    return X, y, subjects, sessions


def normalize_with_stats(X, data_mean, data_std):
    """Normalize data using pre-computed statistics (same as training)."""
    X_norm = ((X - data_mean) / data_std).astype(np.float32)
    X_norm = np.clip(X_norm, -5.0, 5.0)  # Same clip as training
    return X_norm


# ============================================================================
# Utility functions
# ============================================================================

def train_and_eval_classifier(X_train, y_train, X_test, y_test):
    """Train classifier and return accuracy and kappa. No validation split — use training accuracy for model selection (consistent with original evaluation)."""
    clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(clf.parameters(), lr=CLASSIFIER_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CLASSIFIER_EPOCHS)

    X_tr_t = torch.FloatTensor(X_train).to(DEVICE)
    y_tr_t = torch.LongTensor(y_train).to(DEVICE)

    best_acc = 0.0
    best_state = None

    for ep in range(1, CLASSIFIER_EPOCHS + 1):
        clf.train()
        indices = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), CLASSIFIER_BATCH_SIZE):
            batch_idx = indices[i:i + CLASSIFIER_BATCH_SIZE]
            xb = X_tr_t[batch_idx]
            yb = y_tr_t[batch_idx]
            logits = clf(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if ep % 10 == 0 or ep == CLASSIFIER_EPOCHS:
            clf.eval()
            with torch.no_grad():
                all_preds = []
                for j in range(0, len(X_tr_t), CLASSIFIER_BATCH_SIZE):
                    batch_logits = clf(X_tr_t[j:j+CLASSIFIER_BATCH_SIZE])
                    all_preds.append(batch_logits.argmax(1))
                train_pred = torch.cat(all_preds)
                train_acc = (train_pred == y_tr_t).float().mean().item()
            if train_acc > best_acc:
                best_acc = train_acc
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)
    clf.eval()
    with torch.no_grad():
        all_preds = []
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        for j in range(0, len(X_test_t), CLASSIFIER_BATCH_SIZE):
            batch_logits = clf(X_test_t[j:j+CLASSIFIER_BATCH_SIZE])
            all_preds.append(batch_logits.argmax(1))
        pred = torch.cat(all_preds).cpu().numpy()
    acc = accuracy_score(y_test, pred)
    kappa = cohen_kappa_score(y_test, pred)
    del clf
    torch.cuda.empty_cache()
    return acc, kappa


# ============================================================================
# Augmentation methods
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

def load_ddpm_bci2a(checkpoint_path):
    """Load DDPM and return (model, ckpt, data_mean, data_std)."""
    if not os.path.exists(checkpoint_path):
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
        data_mean = ckpt["data_mean"]
        data_std = ckpt["data_std"]
        print(f"  Loaded normalization stats from checkpoint: mean={data_mean.mean():.6f}, std={data_std.mean():.6f}")
    else:
        print(f"  WARNING: checkpoint has no data_mean/data_std, will compute from current data")

    print(f"  Loaded DDPM from {checkpoint_path}")
    return ddpm, ckpt, data_mean, data_std


def load_baseline_bci2a(model_name):
    """Load a trained baseline model for BCI2a."""
    ckpt_dir = os.path.join("checkpoints", "baselines")
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_bci2a.pt')
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
    """Generate samples from baseline models. Output clipped to [-5, 5] to match training data range."""
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


def generate_ddpm_samples(ddpm, n_per_class, guidance_scale=1.0, eta=0.5):
    gen_X, gen_y = [], []
    DDPM_BATCH = 16
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            remaining = n_per_class
            while remaining > 0:
                batch = min(DDPM_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample_ddim(batch, y, steps=50, guidance_scale=guidance_scale, eta=eta, device=str(DEVICE))
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

def evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y):
    X_aug = np.concatenate([X_train, gen_X])
    y_aug = np.concatenate([y_train, gen_y])
    return train_and_eval_classifier(X_aug, y_aug, X_test, y_test)


CACHE_DIR = "outputs/results/cache_bci2a_v2_kappa"
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
        print(f"  [Resume] {scenario}: starting from subject {start_idx + 1} (cached)")
        return all_results, start_idx
    return {name: [] for name in METHOD_NAMES}, 0


# ============================================================================
# Three scenarios
# ============================================================================

def run_within_subject(X, y, subjects, ddpm, baseline_models, guidance_scale, eta=0.5, run_methods=None):
    print("\n" + "=" * 70)
    print("1. BCI2a Within-Subject Test")
    print("=" * 70)

    all_results, start_idx = load_cache("within_subject")
    unique_subjects = np.unique(subjects)

    # Determine which methods to skip (use cached results)
    if run_methods is not None:
        skip_methods = set(METHOD_NAMES) - set(run_methods)
    else:
        skip_methods = set()

    for idx, subj_id in enumerate(unique_subjects):
        if idx < start_idx:
            continue
        # Check if this subject's results are already complete for all run_methods
        if skip_methods and all(len(all_results.get(m, [])) > idx for m in skip_methods):
            # Only run methods that need re-evaluation
            pass
        print(f"\nSubject {idx + 1}/{len(unique_subjects)} (ID={subj_id}):")
        mask = subjects == subj_id
        X_subj, y_subj = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj)

        samples_per_class = int(len(X_train) // NUM_CLASSES)
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Gen: {samples_per_class}x{NUM_CLASSES}")

        # Baseline
        if 'baseline' not in skip_methods or len(all_results.get('baseline', [])) <= idx:
            acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
            if len(all_results.get('baseline', [])) > idx:
                all_results['baseline'][idx] = (acc, kappa)
            else:
                all_results['baseline'].append((acc, kappa))
            print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Baseline: [cached] {all_results['baseline'][idx][0]*100:.2f}%")

        # Gaussian Noise
        if 'gaussian_noise' not in skip_methods or len(all_results.get('gaussian_noise', [])) <= idx:
            gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('gaussian_noise', [])) > idx:
                all_results['gaussian_noise'][idx] = (acc, kappa)
            else:
                all_results['gaussian_noise'].append((acc, kappa))
            print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Gaussian Noise: [cached] {all_results['gaussian_noise'][idx][0]*100:.2f}%")

        # SMOTE
        if 'smote' not in skip_methods or len(all_results.get('smote', [])) <= idx:
            gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('smote', [])) > idx:
                all_results['smote'][idx] = (acc, kappa)
            else:
                all_results['smote'].append((acc, kappa))
            print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  SMOTE: [cached] {all_results['smote'][idx][0]*100:.2f}%")

        # Baseline models
        for name, model in baseline_models.items():
            if name not in skip_methods or len(all_results.get(name, [])) <= idx:
                gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get(name, [])) > idx:
                    all_results[name][idx] = (acc, kappa)
                else:
                    all_results[name].append((acc, kappa))
                print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  {name}: [cached] {all_results[name][idx][0]*100:.2f}%")

        # DDPM
        if ddpm is not None:
            if 'ddpm' not in skip_methods or len(all_results.get('ddpm', [])) <= idx:
                gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale, eta)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get('ddpm', [])) > idx:
                    all_results['ddpm'][idx] = (acc, kappa)
                else:
                    all_results['ddpm'].append((acc, kappa))
                print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  DDPM: [cached] {all_results['ddpm'][idx][0]*100:.2f}%")

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


def run_cross_session(X, y, subjects, sessions, ddpm, baseline_models, guidance_scale, eta=0.5, run_methods=None):
    print("\n" + "=" * 70)
    print("2. BCI2a Cross-Session Test (T->E)")
    print("=" * 70)

    all_results, start_idx = load_cache("cross_session")
    unique_subjects = np.unique(subjects)

    if run_methods is not None:
        skip_methods = set(METHOD_NAMES) - set(run_methods)
    else:
        skip_methods = set()

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
        samples_per_class = int(len(X_train) // NUM_CLASSES)

        # Baseline
        if 'baseline' not in skip_methods or len(all_results.get('baseline', [])) <= idx:
            acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
            if len(all_results.get('baseline', [])) > idx:
                all_results['baseline'][idx] = (acc, kappa)
            else:
                all_results['baseline'].append((acc, kappa))
            print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Baseline: [cached]")

        # Gaussian Noise
        if 'gaussian_noise' not in skip_methods or len(all_results.get('gaussian_noise', [])) <= idx:
            gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('gaussian_noise', [])) > idx:
                all_results['gaussian_noise'][idx] = (acc, kappa)
            else:
                all_results['gaussian_noise'].append((acc, kappa))
            print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Gaussian Noise: [cached]")

        # SMOTE
        if 'smote' not in skip_methods or len(all_results.get('smote', [])) <= idx:
            gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('smote', [])) > idx:
                all_results['smote'][idx] = (acc, kappa)
            else:
                all_results['smote'].append((acc, kappa))
            print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  SMOTE: [cached]")

        # Baseline models
        for name, model in baseline_models.items():
            if name not in skip_methods or len(all_results.get(name, [])) <= idx:
                gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get(name, [])) > idx:
                    all_results[name][idx] = (acc, kappa)
                else:
                    all_results[name].append((acc, kappa))
                print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  {name}: [cached]")

        # DDPM
        if ddpm is not None:
            if 'ddpm' not in skip_methods or len(all_results.get('ddpm', [])) <= idx:
                gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale, eta)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get('ddpm', [])) > idx:
                    all_results['ddpm'][idx] = (acc, kappa)
                else:
                    all_results['ddpm'].append((acc, kappa))
                print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  DDPM: [cached]")

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


def run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, guidance_scale, eta=0.5, run_methods=None):
    print("\n" + "=" * 70)
    print("3. BCI2a Cross-Subject Test (LOSO, T-only)")
    print("=" * 70)

    all_results, start_idx = load_cache("cross_subject")
    unique_subjects = np.unique(subjects)

    if run_methods is not None:
        skip_methods = set(METHOD_NAMES) - set(run_methods)
    else:
        skip_methods = set()

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
        samples_per_class = int(len(X_train) // NUM_CLASSES // 2)  # 0.5x for cross-subject

        # Baseline
        if 'baseline' not in skip_methods or len(all_results.get('baseline', [])) <= idx:
            acc, kappa = train_and_eval_classifier(X_train, y_train, X_test, y_test)
            if len(all_results.get('baseline', [])) > idx:
                all_results['baseline'][idx] = (acc, kappa)
            else:
                all_results['baseline'].append((acc, kappa))
            print(f"  Baseline: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Baseline: [cached]")

        # Gaussian Noise
        if 'gaussian_noise' not in skip_methods or len(all_results.get('gaussian_noise', [])) <= idx:
            gen_X, gen_y = gaussian_noise_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('gaussian_noise', [])) > idx:
                all_results['gaussian_noise'][idx] = (acc, kappa)
            else:
                all_results['gaussian_noise'].append((acc, kappa))
            print(f"  Gaussian Noise: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  Gaussian Noise: [cached]")

        # SMOTE
        if 'smote' not in skip_methods or len(all_results.get('smote', [])) <= idx:
            gen_X, gen_y = smote_augment(X_train, y_train, samples_per_class)
            acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
            if len(all_results.get('smote', [])) > idx:
                all_results['smote'][idx] = (acc, kappa)
            else:
                all_results['smote'].append((acc, kappa))
            print(f"  SMOTE: {acc*100:.2f}%, Kappa: {kappa:.4f}")
        else:
            print(f"  SMOTE: [cached]")

        # Baseline models
        for name, model in baseline_models.items():
            if name not in skip_methods or len(all_results.get(name, [])) <= idx:
                gen_X, gen_y = generate_baseline_samples(model, name, samples_per_class)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get(name, [])) > idx:
                    all_results[name][idx] = (acc, kappa)
                else:
                    all_results[name].append((acc, kappa))
                print(f"  {name}: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  {name}: [cached]")

        # DDPM
        if ddpm is not None:
            if 'ddpm' not in skip_methods or len(all_results.get('ddpm', [])) <= idx:
                gen_X, gen_y = generate_ddpm_samples(ddpm, samples_per_class, guidance_scale, eta)
                acc, kappa = evaluate_augmentation(X_train, y_train, X_test, y_test, gen_X, gen_y)
                if len(all_results.get('ddpm', [])) > idx:
                    all_results['ddpm'][idx] = (acc, kappa)
                else:
                    all_results['ddpm'].append((acc, kappa))
                print(f"  DDPM: {acc*100:.2f}%, Kappa: {kappa:.4f}")
            else:
                print(f"  DDPM: [cached]")

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
    parser = argparse.ArgumentParser(description="BCI2a All Methods Evaluation")
    parser.add_argument("--data_root", type=str, default="data/processed/BCI2a")
    parser.add_argument("--ddpm_ckpt", type=str, default="checkpoints/best_class_discriminative.pt")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.5, help="DDIM stochasticity (0=deterministic, 1=DDPM)")
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
    print("BCI2a All Methods Three-Scenario Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load raw data (no normalization)
    X_raw, y, subjects, sessions = load_bci2a_data(args.data_root)

    # Only load DDPM if needed
    ddpm = None
    if run_methods is None or 'ddpm' in run_methods:
        ddpm, ckpt, data_mean, data_std = load_ddpm_bci2a(args.ddpm_ckpt)
    else:
        # Still need normalization stats
        ckpt_path = args.ddpm_ckpt
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            data_mean = ckpt.get('data_mean') if isinstance(ckpt, dict) else None
            data_std = ckpt.get('data_std') if isinstance(ckpt, dict) else None
        else:
            data_mean, data_std = None, None

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

    # Only load baseline models that are needed
    baseline_models = {}
    needed_baselines = ['cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth']
    if run_methods is not None:
        needed_baselines = [n for n in needed_baselines if n in run_methods]
    for name in needed_baselines:
        model = load_baseline_bci2a(name)
        if model is not None:
            baseline_models[name] = model
            print(f"  Loaded {name} for BCI2a")

    # Run three scenarios
    results = {}
    results['within_subject'] = run_within_subject(X, y, subjects, ddpm, baseline_models, args.guidance_scale, args.eta, run_methods)
    results['cross_session'] = run_cross_session(X, y, subjects, sessions, ddpm, baseline_models, args.guidance_scale, args.eta, run_methods)
    results['cross_subject'] = run_cross_subject(X, y, subjects, sessions, ddpm, baseline_models, args.guidance_scale, args.eta, run_methods)

    # Print summary
    print("\n" + "=" * 70)
    print("BCI2a Final Results Summary")
    print("=" * 70)
    for scenario, methods in results.items():
        print(f"\n{scenario}:")
        for method, vals in methods.items():
            print(f"  {method:<20}: Acc {vals['acc_mean']*100:.2f}% +/- {vals['acc_std']*100:.2f}%, Kappa {vals['kappa_mean']:.4f} +/- {vals['kappa_std']:.4f}")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    out_path = "outputs/results/all_methods_bci2a_kappa.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
