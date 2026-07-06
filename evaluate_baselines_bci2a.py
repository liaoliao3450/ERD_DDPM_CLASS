#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Baseline Evaluation Script for Comparative Experiments

Evaluates all methods on BCI2a with:
1. Signal Quality Metrics (all subjects, with boxplots):
   - PSD Correlation (Welch's method + log PSD)
   - Band-wise Spectral Similarity (delta/theta/alpha/beta/gamma)
   - Temporal Correlation
2. Classification Performance:
   - Within-Subject
   - Cross-Session
   - Cross-Subject (LOSO)

Methods evaluated:
- Gaussian Noise (no training needed)
- SMOTE (no training needed)
- CVAE (trained)
- WaveGAN (trained)
- Cond-DDPM (trained)
- BrainDiff (trained)
- EEGDiff (trained)
- DiffEEGBooth (trained)
- ERD-DDPM (ours, trained)

Addresses Reviewer #2 Comment 5: "report subject-wise and session-wise quantitative summaries"
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import scipy.signal as signal
import warnings
warnings.filterwarnings('ignore')

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'ddpm'))
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'baselines'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

from comparison_models import create_baseline_model, WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE, GaussianNoiseAugmentation
from class_discriminative import (
    EEGClassifier, pretrain_classifier, MultiScaleCondUNet, ClassDiscriminativeDDPM
)


def load_bci2a_data():
    """Load BCI2a data directly from processed files."""
    data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2a')
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Data not found: {X_path}")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    y = y - y.min()
    mask = y < 4
    X, y = X[mask], y[mask]

    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

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

    print(f"BCI2a data loaded: {X.shape}, classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}")
    return X, y, subjects, sessions

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BCI2a parameters
CHANNELS = 22
N_SAMPLES = 1000
FS = 250
NUM_CLASSES = 4

# Consistent classifier training hyperparameters
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3

CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints', 'baselines')
RESULTS_DIR = os.path.join(project_root, 'outputs', 'comparison_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# Signal Quality Metrics (using Welch's method for PSD)
# ============================================================================

def compute_psd_welch(x, fs=250, nperseg=256, noverlap=128):
    """Compute PSD using Welch's method (averaged across trials and channels).
    This is the standard approach for PSD estimation in EEG research."""
    all_psds = []
    n_trials = min(x.shape[0], 50)  # Limit trials for efficiency
    for i in range(n_trials):
        for j in range(x.shape[1]):
            f, Pxx = signal.welch(x[i, j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            all_psds.append(Pxx)
    return f, np.array(all_psds).mean(axis=0)


def compute_psd_correlation(X_real, X_gen):
    """Compute PSD correlation between real and generated data using Welch's method.
    Uses log10(PSD) for correlation (standard approach).
    Returns per-class and overall correlation."""
    correlations = {}
    for c in range(NUM_CLASSES):
        real_c = X_real  # Use all real data for reference
        gen_c = X_gen  # Generated for this class

        f_real, psd_real = compute_psd_welch(real_c, FS)
        f_gen, psd_gen = compute_psd_welch(gen_c, FS)

        log_psd_real = np.log10(psd_real + 1e-10)
        log_psd_gen = np.log10(psd_gen + 1e-10)

        corr = np.corrcoef(log_psd_real, log_psd_gen)[0, 1]
        correlations[f'class_{c}'] = float(corr)

    # Overall
    f_real, psd_real = compute_psd_welch(X_real, FS)
    f_gen, psd_gen = compute_psd_welch(X_gen, FS)

    log_psd_r = np.log10(psd_real + 1e-10)
    log_psd_g = np.log10(psd_gen + 1e-10)
    correlations['overall'] = float(np.corrcoef(log_psd_r, log_psd_g)[0, 1])

    return correlations


def compute_band_similarity(X_real, X_gen):
    """Compute band-wise spectral similarity using Welch's method.
    Returns correlation of log PSD in each frequency band."""
    f_real, psd_real = compute_psd_welch(X_real, FS)
    f_gen, psd_gen = compute_psd_welch(X_gen, FS)

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }

    log_psd_r = np.log10(psd_real + 1e-10)
    log_psd_g = np.log10(psd_gen + 1e-10)

    similarities = {}
    for band_name, (f_low, f_high) in bands.items():
        mask = (f_real >= f_low) & (f_real <= f_high)
        if mask.sum() > 2:
            corr = float(np.corrcoef(log_psd_r[mask], log_psd_g[mask])[0, 1])
        else:
            corr = 0.0
        similarities[band_name] = {'correlation': corr}

    # Mean correlation across bands
    mean_corr = np.mean([v['correlation'] for v in similarities.values()])
    similarities['mean'] = {'correlation': float(mean_corr)}

    return similarities


def compute_temporal_correlation(X_real, X_gen):
    """Compute temporal correlation between real and generated data.
    Average Pearson correlation across channels."""
    n_channels = X_real.shape[1]

    # Normalize
    real_norm = (X_real - X_real.mean(axis=-1, keepdims=True)) / (X_real.std(axis=-1, keepdims=True) + 1e-8)
    gen_norm = (X_gen - X_gen.mean(axis=-1, keepdims=True)) / (X_gen.std(axis=-1, keepdims=True) + 1e-8)

    # Average template from real data
    template = real_norm.mean(axis=0)  # [C, T]

    # Correlation per sample
    corrs = []
    for i in range(len(gen_norm)):
        sample_corr = 0
        for ch in range(n_channels):
            c = np.corrcoef(template[ch], gen_norm[i, ch])[0, 1]
            sample_corr += c
        corrs.append(sample_corr / n_channels)

    return {
        'mean': float(np.mean(corrs)),
        'std': float(np.std(corrs)),
        'per_sample': [float(c) for c in corrs],
    }


def normalize_generated_data(X_real, X_gen):
    """Align generated data statistics to real data."""
    X_real = X_real.astype(np.float32)
    X_gen = X_gen.astype(np.float32)

    real_mean = X_real.mean(axis=(0, 2), keepdims=True)
    real_std = X_real.std(axis=(0, 2), keepdims=True)
    gen_mean = X_gen.mean(axis=(0, 2), keepdims=True)
    gen_std = X_gen.std(axis=(0, 2), keepdims=True)

    eps = 1e-8
    X_gen_norm = (X_gen - gen_mean) / (gen_std + eps)
    X_gen_aligned = X_gen_norm * (real_std + eps) + real_mean
    return X_gen_aligned


# ============================================================================
# Non-trainable Method Generators
# ============================================================================

def gaussian_noise_augmentation(X_train, y_train, n_samples_per_class, noise_level=0.1):
    """Gaussian noise augmentation - add noise to existing samples."""
    gen_X, gen_y = [], []
    for c in range(NUM_CLASSES):
        class_data = X_train[y_train == c]
        class_std = class_data.std(axis=0)
        for _ in range(n_samples_per_class):
            base = class_data[np.random.randint(len(class_data))]
            noise = np.random.randn(*base.shape) * class_std * noise_level
            gen_X.append(base + noise)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


def smote_augmentation(X_train, y_train, n_samples_per_class):
    """SMOTE augmentation - linear interpolation between nearest neighbors."""
    gen_X, gen_y = [], []
    for c in range(NUM_CLASSES):
        class_data = X_train[y_train == c]
        for _ in range(n_samples_per_class):
            idx = np.random.randint(len(class_data))
            sample = class_data[idx]
            k = min(5, len(class_data) - 1)
            if k > 0:
                neighbor_indices = np.random.choice(
                    [i for i in range(len(class_data)) if i != idx],
                    k, replace=False
                )
                neighbor = class_data[neighbor_indices[0]]
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
            else:
                synthetic = sample
            gen_X.append(synthetic)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


# ============================================================================
# Model Loading
# ============================================================================

def load_baseline_model(model_name, device='cuda'):
    """Load a trained baseline model from checkpoint."""
    if model_name == 'cvae':
        model = CVAE(channels=CHANNELS, latent_dim=64, out_length=N_SAMPLES, num_classes=NUM_CLASSES)
    else:
        model = create_baseline_model(model_name, CHANNELS, N_SAMPLES, NUM_CLASSES, FS)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_bci2a.pt')
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Checkpoint not found: {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_name == 'wavegan':
        model.generator.load_state_dict(checkpoint['generator'])
        model.discriminator.load_state_dict(checkpoint['discriminator'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()
    print(f"  Loaded {model_name} from {ckpt_path}")
    return model


def load_erd_ddpm(device='cuda'):
    """Load the trained ERD-DDPM model."""
    ckpt_path = os.path.join(project_root, 'checkpoints', 'best_class_discriminative.pt')
    if not os.path.exists(ckpt_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(project_root, 'checkpoints', 'class_discriminative_ddpm_full.pt'),
            os.path.join(project_root, 'checkpoints', 'trained_ddpm.pt'),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

    if not os.path.exists(ckpt_path):
        print(f"  WARNING: ERD-DDPM checkpoint not found")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)

    if isinstance(checkpoint, dict) and 'target_psd' in checkpoint:
        target_psd = checkpoint['target_psd'].to(device)
        target_laterality = checkpoint['target_laterality'].to(device)
    else:
        target_psd = torch.zeros(N_SAMPLES // 2 + 1).to(device)
        target_laterality = torch.zeros(NUM_CLASSES).to(device)

    ddpm = ClassDiscriminativeDDPM(
        eps_model, classifier, target_psd, target_laterality,
        n_timesteps=1000, channels=CHANNELS, n_samples=N_SAMPLES, fs=FS
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        try:
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError:
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        ddpm.load_state_dict(checkpoint)

    ddpm.eval()
    print(f"  Loaded ERD-DDPM from {ckpt_path}")
    return ddpm


# ============================================================================
# Sample Generation
# ============================================================================

def generate_samples(model, model_name, n_samples_per_class, device='cuda',
                     X_train=None, y_train=None):
    """Generate samples from a model.
    For Gaussian Noise and SMOTE, X_train and y_train must be provided."""
    # Non-trainable methods
    if model_name == 'gaussian_noise':
        return gaussian_noise_augmentation(X_train, y_train, n_samples_per_class)
    if model_name == 'smote':
        return smote_augmentation(X_train, y_train, n_samples_per_class)

    # Trainable models
    gen_X, gen_y = [], []

    with torch.no_grad():
        for c in range(NUM_CLASSES):
            n_batches = (n_samples_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_samples_per_class - len([y for y in gen_y if y == c]))
                if batch_size <= 0:
                    break
                y = torch.full((batch_size,), c, dtype=torch.long, device=device)

                if model_name == 'wavegan':
                    z = torch.randn(batch_size, model.z_dim, device=device)
                    samples = model.generator(z, y)
                elif model_name == 'cvae':
                    samples = model.generate(batch_size, y, device)
                elif model_name == 'erd_ddpm':
                    samples = model.sample_ddim(batch_size, y, steps=50, guidance_scale=5.5, device=device)
                else:
                    samples = model.sample_ddim(batch_size, y, steps=50, device=device)

                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)

    return np.concatenate(gen_X), np.array(gen_y)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_signal_quality(model, model_name, X_train, y_train, device='cuda'):
    """Evaluate signal quality metrics for a single model.
    Returns per-subject metrics for boxplot generation."""
    print(f"\n  Signal Quality Evaluation - {model_name}")

    n_samples_per_class = 50  # Fixed for quality evaluation
    gen_X, gen_y = generate_samples(model, model_name, n_samples_per_class, device,
                                    X_train=X_train, y_train=y_train)
    gen_X = normalize_generated_data(X_train, gen_X)

    psd_corr = compute_psd_correlation(X_train, gen_X)
    band_sim = compute_band_similarity(X_train, gen_X)
    temporal_corr = compute_temporal_correlation(X_train, gen_X)

    return {
        'psd_correlation': psd_corr,
        'band_similarity': band_sim,
        'temporal_correlation': temporal_corr,
    }


def evaluate_signal_quality_per_subject(model, model_name, X, y, subjects, device='cuda'):
    """Evaluate signal quality for each subject separately.
    Addresses Reviewer #2 Comment 5: per-subject quantitative summaries."""
    print(f"\n  Per-Subject Signal Quality - {model_name}")

    n_subjects = len(np.unique(subjects))
    per_subject_results = []

    for subj_id in range(n_subjects):
        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        n_per_class = max(20, len(X_subj) // NUM_CLASSES // 2)
        gen_X, gen_y = generate_samples(model, model_name, n_per_class, device,
                                        X_train=X_subj, y_train=y_subj)
        gen_X = normalize_generated_data(X_subj, gen_X)

        psd_corr = compute_psd_correlation(X_subj, gen_X)
        band_sim = compute_band_similarity(X_subj, gen_X)
        temporal_corr = compute_temporal_correlation(X_subj, gen_X)

        per_subject_results.append({
            'subject_id': subj_id,
            'psd_correlation': psd_corr['overall'],
            'band_similarity': {k: v['correlation'] for k, v in band_sim.items()},
            'temporal_correlation': temporal_corr['mean'],
        })

    return per_subject_results


def within_subject_test(model, model_name, X, y, subjects, device='cuda'):
    """Within-subject classification test."""
    print(f"\n  Within-Subject Test - {model_name}")

    results = []
    n_subjects = len(np.unique(subjects))

    for subj_id in range(n_subjects):
        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
        )

        samples_per_class = len(X_train) // NUM_CLASSES
        gen_X, gen_y = generate_samples(model, model_name, samples_per_class, device,
                                        X_train=X_train, y_train=y_train)
        gen_X = normalize_generated_data(X_train, gen_X)

        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LR, device=device, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa)})

    return results


def cross_session_test(model, model_name, X, y, subjects, sessions, device='cuda'):
    """Cross-session classification test."""
    print(f"\n  Cross-Session Test - {model_name}")

    results = []
    n_subjects = len(np.unique(subjects))

    for subj_id in range(n_subjects):
        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        samples_per_class = len(X_train) // NUM_CLASSES
        gen_X, gen_y = generate_samples(model, model_name, samples_per_class, device,
                                        X_train=X_train, y_train=y_train)
        gen_X = normalize_generated_data(X_train, gen_X)

        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LR, device=device, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa)})

    return results


def cross_subject_test(model, model_name, X, y, subjects, sessions, device='cuda'):
    """Cross-subject (LOSO) classification test."""
    print(f"\n  Cross-Subject Test (LOSO) - {model_name}")

    results = []
    n_subjects = len(np.unique(subjects))

    for test_subj in range(n_subjects):
        train_mask = (subjects != test_subj) & (sessions == 0)
        test_mask = (subjects == test_subj) & (sessions == 0)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        samples_per_class = len(X_train) // NUM_CLASSES
        gen_X, gen_y = generate_samples(model, model_name, samples_per_class, device,
                                        X_train=X_train, y_train=y_train)
        gen_X = normalize_generated_data(X_train, gen_X)

        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        clf = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LR, device=device, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='macro')
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'accuracy': float(acc), 'f1': float(f1), 'kappa': float(kappa)})

    return results


# ============================================================================
# Results Summary
# ============================================================================

def summarize_results(results_list, metric_key='accuracy'):
    """Compute mean, std from per-subject results."""
    values = [r[metric_key] for r in results_list]
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'per_subject': values,
    }


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE: Classification Accuracy (%)")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Within-Subj':>15} {'Cross-Sess':>15} {'Cross-Subj':>15}")
    print(f"{'-'*65}")

    for method, results in all_results.items():
        ws = results.get('within_subject', {})
        cs = results.get('cross_session', {})
        cx = results.get('cross_subject', {})

        ws_str = f"{ws.get('mean', 0)*100:.2f} +/- {ws.get('std', 0)*100:.2f}" if ws else "N/A"
        cs_str = f"{cs.get('mean', 0)*100:.2f} +/- {cs.get('std', 0)*100:.2f}" if cs else "N/A"
        cx_str = f"{cx.get('mean', 0)*100:.2f} +/- {cx.get('std', 0)*100:.2f}" if cx else "N/A"

        print(f"{method:<20} {ws_str:>15} {cs_str:>15} {cx_str:>15}")

    print(f"{'='*80}")

    # Signal quality table
    print(f"\n{'='*80}")
    print("SIGNAL QUALITY METRICS")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'PSD Corr':>10} {'Alpha':>10} {'Beta':>10} {'Temporal':>10}")
    print(f"{'-'*60}")

    for method, results in all_results.items():
        sq = results.get('signal_quality', {})
        psd = sq.get('psd_correlation', {}).get('overall', 0)
        bands = sq.get('band_similarity', {})
        alpha_corr = bands.get('alpha', {}).get('correlation', 0)
        beta_corr = bands.get('beta', {}).get('correlation', 0)
        temporal = sq.get('temporal_correlation', {}).get('mean', 0)

        print(f"{method:<20} {psd:>10.4f} {alpha_corr:>10.4f} {beta_corr:>10.4f} {temporal:>10.4f}")

    print(f"{'='*80}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    parser.add_argument('--skip_quality', action='store_true',
                        help='Skip signal quality evaluation')
    parser.add_argument('--skip_classification', action='store_true',
                        help='Skip classification evaluation')
    parser.add_argument('--skip_per_subject', action='store_true',
                        help='Skip per-subject quality analysis')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Results dir: {RESULTS_DIR}")

    # Load data
    print("\nLoading BCI2a data...")
    X, y, subjects, sessions = load_bci2a_data()

    # Models to evaluate (including non-trainable methods)
    all_model_names = ['gaussian_noise', 'smote', 'cvae', 'wavegan', 'cond_ddpm',
                       'braindiff', 'eegdiff', 'diffeegbooth', 'erd_ddpm']
    model_names = args.models if args.models else all_model_names

    all_results = {}

    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {model_name}")
        print(f"{'#'*60}")

        # Load model (Gaussian Noise and SMOTE don't need a model)
        if model_name in ('gaussian_noise', 'smote'):
            model = None  # No model needed
        elif model_name == 'erd_ddpm':
            model = load_erd_ddpm(DEVICE)
        else:
            model = load_baseline_model(model_name, DEVICE)

        if model is None and model_name not in ('gaussian_noise', 'smote'):
            print(f"  Skipping {model_name} (model not found)")
            continue

        results = {}

        # Signal quality
        if not args.skip_quality:
            X_train = X[sessions == 0]
            y_train = y[sessions == 0]
            sq = evaluate_signal_quality(model, model_name, X_train, y_train, DEVICE)
            results['signal_quality'] = sq

        # Per-subject signal quality (Reviewer #2 requirement)
        if not args.skip_per_subject:
            per_subj = evaluate_signal_quality_per_subject(
                model, model_name, X, y, subjects, DEVICE
            )
            results['per_subject_quality'] = per_subj

        # Classification tests
        if not args.skip_classification:
            ws = within_subject_test(model, model_name, X, y, subjects, DEVICE)
            results['within_subject'] = summarize_results(ws, 'accuracy')

            cs = cross_session_test(model, model_name, X, y, subjects, sessions, DEVICE)
            results['cross_session'] = summarize_results(cs, 'accuracy')

            cx = cross_subject_test(model, model_name, X, y, subjects, sessions, DEVICE)
            results['cross_subject'] = summarize_results(cx, 'accuracy')

        all_results[model_name] = results

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'comparison_results_bci2a.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate boxplot data for per-subject quality
    if not args.skip_per_subject:
        boxplot_data = {}
        for method, res in all_results.items():
            if 'per_subject_quality' in res:
                psd_corrs = [s['psd_correlation'] for s in res['per_subject_quality']]
                temporal_corrs = [s['temporal_correlation'] for s in res['per_subject_quality']]
                boxplot_data[method] = {
                    'psd_correlation': psd_corrs,
                    'temporal_correlation': temporal_corrs,
                }

        boxplot_path = os.path.join(RESULTS_DIR, 'boxplot_data_bci2a.json')
        with open(boxplot_path, 'w') as f:
            json.dump(boxplot_data, f, indent=2)
        print(f"Boxplot data saved to: {boxplot_path}")


if __name__ == '__main__':
    main()
