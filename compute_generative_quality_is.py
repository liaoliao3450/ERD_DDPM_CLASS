#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive generative quality metrics for all augmentation methods
across BCI2a, BCI2b, and PhysioNet MI4C datasets.

Metrics: IS, FID, sFID, Precision, Recall
- IS: Inception Score (higher = better class separability)
- FID: Frechet Inception Distance on EEGNet pooled features (lower = better)
- sFID: spatial FID on EEGNet spatial features (lower = better)
- Precision: sample quality (higher = better)
- Recall: sample diversity (higher = better)

References:
- IS [36]: Salimans et al., 2016
- FID [37]: Heusel et al., 2017
- sFID [38]: Spectral FID (frequency-domain FID)
- Precision/Recall [39]: Kynkaanniemi et al., 2019 (via prdc library)
- EEGNet [41]: Lawhern et al., 2018 (pretrained for feature extraction)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

try:
    from umap import UMAP
except ImportError:
    try:
        from sklearn.manifold import UMAP
    except ImportError:
        UMAP = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'baselines'))
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'ddpm'))

from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE
from class_discriminative import MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(project_root, 'checkpoints')
OUTPUT_DIR = os.path.join(project_root, 'outputs', 'results', 'quality_metrics')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import EEGNet for feature extraction
from core.classifiers.eegnet import EEGNet


# ============================================================================
# EEGNet-based Feature Extractor
# ============================================================================
class EEGNetFeatureExtractor(nn.Module):
    """EEGNet-based feature extractor for metric computation.

    Uses pretrained EEGNet [41] as specified in the paper.
    """

    def __init__(self, channels, n_samples, num_classes):
        super().__init__()
        self.backbone = EEGNet(
            n_channels=channels,
            n_samples=n_samples,
            n_classes=num_classes,
        )
        self.num_features = self.backbone.feature_dim

    def forward(self, x):
        """Returns (logits, features)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.backbone._forward_features(x)
        logits = self.backbone.classifier(features)
        return logits, features

    def get_features(self, x):
        """Get globally pooled features for FID computation."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.backbone._forward_features(x)

    def get_spatial_features(self, x):
        """Get spatial features (before final avg pooling) for sFID computation.

        Returns features after separable conv but before the final avgpool2,
        preserving spatio-temporal structure.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(1)
            h = self.backbone.conv1(x)
            h = self.backbone.batchnorm1(h)
            h = self.backbone.conv2(h)
            h = self.backbone.batchnorm2(h)
            h = self.backbone.elu1(h)
            h = self.backbone.avgpool1(h)
            h = self.backbone.dropout1(h)
            h = self.backbone.sep_depthwise(h)
            h = self.backbone.sep_pointwise(h)
            h = self.backbone.batchnorm3(h)
            h = self.backbone.elu2(h)
            # Stop before avgpool2 - flatten to preserve spatial structure
            spatial_feat = h.flatten(1)
        return spatial_feat


def pretrain_feature_extractor(channels, n_samples, num_classes, X_train, y_train, epochs=300):
    """Pre-train EEGNet feature extractor on real data.

    Key design choices for FID-compatible feature extraction:
    - NO data augmentation: we want maximum discriminative features
    - NO label smoothing: sharp class boundaries give better IS/FID
    - High lr + cosine annealing for strong fitting
    - Train on ALL data for best feature quality
    """
    model = EEGNetFeatureExtractor(channels, n_samples, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train).to(DEVICE)
    batch_size = 64

    for ep in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = indices[i:i + batch_size]
            logits, _ = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_t)[0].argmax(1) == y_t).float().mean().item()
    print(f"  Feature extractor accuracy: {acc:.4f}")
    return model


# ============================================================================
# Metric Computation
# ============================================================================
def compute_inception_score(probs, num_classes, splits=10):
    """Compute Inception Score from classifier softmax probabilities.

    IS measures class separability: higher IS means generated samples
    have more discriminative class predictions.
    """
    scores = []
    N = probs.shape[0]
    split_size = max(N // splits, 1)
    for k in range(splits):
        part = probs[k * split_size: (k + 1) * split_size]
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores)), float(np.std(scores))


def _cov_regularized(features, reg=0.01):
    """Compute regularized covariance matrix.

    Adds a small multiple of identity to ensure positive-definiteness,
    which is essential for stable matrix square root in FID computation.
    """
    sigma = np.cov(features, rowvar=False)
    if sigma.ndim < 2:
        sigma = np.atleast_2d(sigma)
    sigma += reg * np.eye(sigma.shape[0])
    return sigma


def _normalize_features(real_features, gen_features):
    """Normalize features using real data statistics (zero mean, unit variance).

    This ensures features are on a comparable scale across methods,
    making FID values more interpretable and stable.
    """
    mu = np.mean(real_features, axis=0, keepdims=True)
    std = np.std(real_features, axis=0, keepdims=True).clip(min=1e-6)
    real_norm = (real_features - mu) / std
    gen_norm = (gen_features - mu) / std
    return real_norm, gen_norm


def compute_fid(real_features, gen_features):
    """Compute FID using EEGNet's globally pooled features with PCA.

    PCA stabilizes covariance estimation for reliable FID computation.
    Lower FID = generated distribution closer to real distribution.
    Uses covariance regularization for numerical stability.
    """
    from sklearn.decomposition import PCA

    combined = np.vstack([real_features, gen_features])
    n_components = min(64, len(real_features) - 2, len(gen_features) - 2, combined.shape[1])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)
    real_pca = combined_pca[:len(real_features)]
    gen_pca = combined_pca[len(real_features):]

    mu_r = np.mean(real_pca, axis=0)
    sigma_r = _cov_regularized(real_pca)
    mu_g = np.mean(gen_pca, axis=0)
    sigma_g = _cov_regularized(gen_pca)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(max(0, fid))


def compute_sfid(feat_extractor, real_eeg, gen_eeg):
    """Compute sFID (spatial FID) using EEGNet's spatial features.

    sFID extends FID by considering spatial structure in the feature space,
    making it more suitable for EEG signals where channel topology matters.
    Uses PCA + covariance regularization for stability.
    """
    from sklearn.decomposition import PCA

    batch_size = 128
    real_spatial_list = []
    for i in range(0, len(real_eeg), batch_size):
        xb = torch.FloatTensor(real_eeg[i:i + batch_size]).to(DEVICE)
        real_spatial_list.append(feat_extractor.get_spatial_features(xb).cpu().numpy())
    real_spatial = np.vstack(real_spatial_list)

    gen_spatial_list = []
    for i in range(0, len(gen_eeg), batch_size):
        xb = torch.FloatTensor(gen_eeg[i:i + batch_size]).to(DEVICE)
        gen_spatial_list.append(feat_extractor.get_spatial_features(xb).cpu().numpy())
    gen_spatial = np.vstack(gen_spatial_list)

    # PCA dimensionality reduction
    combined = np.vstack([real_spatial, gen_spatial])
    n_components = min(64, len(real_spatial) - 2, len(gen_spatial) - 2, combined.shape[1])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)
    real_pca = combined_pca[:len(real_spatial)]
    gen_pca = combined_pca[len(real_spatial):]

    mu_r = np.mean(real_pca, axis=0)
    sigma_r = _cov_regularized(real_pca)
    mu_g = np.mean(gen_pca, axis=0)
    sigma_g = _cov_regularized(gen_pca)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    sfid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(max(0, sfid))


def compute_precision_recall(real_features, gen_features, k=3):
    """Compute Precision and Recall (Kynkaanniemi et al. 2019 [39]).

    Uses prdc library with normalized EEGNet features.
    Precision: fraction of generated samples that are close to real data (quality).
    Recall: fraction of real data that is close to generated data (diversity).
    """
    from prdc import compute_prdc
    n = min(len(real_features), len(gen_features), 5000)
    real_f = real_features[:n]
    gen_f = gen_features[:n]
    # Normalize features for stable k-NN distance computation
    real_norm, gen_norm = _normalize_features(real_f, gen_f)
    print(f"Num real: {len(real_norm)} Num fake: {len(gen_norm)}")
    results = compute_prdc(real_features=real_norm, fake_features=gen_norm, nearest_k=k)
    return float(results['precision']), float(results['recall'])


# ============================================================================
# Data Loading
# ============================================================================
def load_dataset(dataset):
    """Load dataset and apply z-score normalization (matching training pipeline)."""
    if dataset == 'bci2a':
        data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2a')
        # 乘以 1e6 转换为微伏，与 train_class_discriminative_ddpm.py 对齐
        # 避免小数值 (1e-6 量级) 导致 z-score 参数数值不稳定
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32) * 1e6
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
        y = y - y.min()
        X, y = X[y < 4], y[y < 4]
        channels, n_samples, fs, num_classes = 22, 1000, 250, 4
    elif dataset == 'bci2b':
        data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2b')
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
        y = y - y.min()
        X, y = X[y < 2], y[y < 2]
        channels, n_samples, fs, num_classes = 3, 1000, 250, 2
    elif dataset == 'physionet':
        data_dir = os.path.join(project_root, 'data', 'processed', 'PhysioNetMI4C')
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
        y = y - y.min()
        channels, n_samples, fs, num_classes = 64, 640, 375, 4
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Per-channel z-score normalization (matching DDPM training: train_class_discriminative_ddpm.py)
    data_mean = X.mean(axis=(0, 2), keepdims=True)
    data_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = ((X - data_mean) / data_std).astype(np.float32)
    # Clip to [-5, 5] matching DDPM training preprocessing
    X = np.clip(X, -5.0, 5.0).astype(np.float32)
    print(f"{dataset}: {X.shape}, range=[{X.min():.2f}, {X.max():.2f}], classes={num_classes}")
    return X, y, channels, n_samples, fs, num_classes, data_mean, data_std


# ============================================================================
# Model Loading and Sample Generation
# ============================================================================
def load_ddpm(dataset, channels, n_samples, fs, num_classes):
    """Load DDPM model for given dataset."""
    ckpt_map = {
        'bci2a': 'best_class_discriminative.pt',
        'bci2b': 'bci2b/trained_ddpm.pt',
        'physionet': 'best_class_discriminative_physionet_mi4c.pt',
    }
    ckpt_path = os.path.join(CKPT_DIR, ckpt_map[dataset])
    if not os.path.exists(ckpt_path):
        print(f"  No DDPM checkpoint: {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    eps_model = MultiScaleCondUNet(channels=channels, num_classes=num_classes).to(DEVICE)
    classifier = EEGClassifier(channels=channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)

    target_psd = ckpt.get("target_psd", torch.zeros(n_samples // 2 + 1, device=DEVICE)).to(DEVICE)
    target_lat = ckpt.get("target_laterality", torch.zeros(num_classes, device=DEVICE)).to(DEVICE)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_lat,
        n_timesteps=1000, channels=channels, n_samples=n_samples, fs=fs,
    ).to(DEVICE)

    if "model_state_dict" in ckpt:
        ddpm.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        ddpm.load_state_dict(ckpt, strict=False)
    ddpm.eval()

    data_mean = ckpt.get("data_mean", None)
    data_std = ckpt.get("data_std", None)
    return ddpm, (data_mean, data_std)


def load_baseline(model_name, dataset, channels, n_samples, fs, num_classes):
    """Load baseline model."""
    ckpt_path = os.path.join(CKPT_DIR, 'baselines', f'{model_name}_{dataset}.pt')
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if model_name == 'cvae':
        model = CVAE(channels=channels, latent_dim=64, out_length=n_samples, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'wavegan':
        model = WaveGAN(channels=channels, out_length=n_samples, num_classes=num_classes)
        model.generator.load_state_dict(ckpt['generator'])
    elif model_name == 'cond_ddpm':
        model = CondDDPM(channels=channels, n_samples=n_samples, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'braindiff':
        model = BrainDiff(channels=channels, n_samples=n_samples, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'eegdiff':
        model = EEGDiff(channels=channels, n_samples=n_samples, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'diffeegbooth':
        model = DiffEEGBooth(channels=channels, n_samples=n_samples, num_classes=num_classes, fs=fs)
        sd = ckpt['model_state_dict']
        if 'target_laterality' in sd and sd['target_laterality'].shape[0] != num_classes:
            sd['target_laterality'] = torch.zeros(num_classes)
        model.load_state_dict(sd, strict=True)
    else:
        return None

    return model.to(DEVICE).eval()


def generate_samples(model, model_name, n_per_class, num_classes,
                     data_mean=None, data_std=None, guidance_scale=0.0):
    """Generate samples from a model.

    All models output z-score normalized data. We clip extreme values
    for numerical stability.

    Args:
        guidance_scale: DDPM classifier guidance strength. 0.0 = no guidance
            (better FID/distribution match), 1.0+ = stronger class separation
            but distribution shift. Default 0.0 for distribution metrics.
    """
    gen_X, gen_y = [], []
    GEN_BATCH = 16

    with torch.no_grad():
        for c in range(num_classes):
            remaining = n_per_class
            while remaining > 0:
                batch = min(GEN_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)

                if model_name == 'wavegan':
                    z = torch.randn(batch, model.z_dim, device=DEVICE)
                    samples = model.generator(z, y)
                elif model_name == 'cvae':
                    samples = model.generate(batch, y, DEVICE)
                elif model_name == 'ddpm':
                    samples = model.sample(batch, y, guidance_scale=guidance_scale, device=str(DEVICE))
                elif model_name in ('cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth'):
                    samples = model.sample(batch, y, device=DEVICE)
                else:
                    continue

                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()

    gen_X = np.concatenate(gen_X)
    gen_y = np.array(gen_y)

    # Clip to [-5, 5] matching DDPM training preprocessing
    gen_X = np.clip(gen_X, -5.0, 5.0).astype(np.float32)

    print(f"  Generated range: [{gen_X.min():.2f}, {gen_X.max():.2f}], mean={gen_X.mean():.4f}, std={gen_X.std():.4f}")

    return gen_X, gen_y


def gaussian_noise_augment(X_train, y_train, n_per_class, num_classes):
    """Gaussian noise augmentation baseline."""
    gen_X, gen_y = [], []
    for c in range(num_classes):
        class_data = X_train[y_train == c]
        class_std = class_data.std(axis=0)
        for _ in range(n_per_class):
            base = class_data[np.random.randint(len(class_data))]
            noise = np.random.randn(*base.shape) * class_std * 0.1
            gen_X.append(base + noise)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


def smote_augment(X_train, y_train, n_per_class, num_classes):
    """SMOTE augmentation baseline."""
    gen_X, gen_y = [], []
    for c in range(num_classes):
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
# UMAP Visualization (DTTD-style)
# ============================================================================
def plot_umap(real_features, all_gen_features, method_names, real_labels,
              all_gen_labels, save_path, dataset_name='Dataset'):
    """Plot UMAP distribution comparison (DTTD-style).

    Layout: 1 row, (1 + n_methods) columns
    - Col 0: Real data (circles, class-colored)
    - Col 1..N: Each method overlaying Real (circles) + Generated (triangles)
    """
    if UMAP is None:
        print("  UMAP not available, skipping UMAP plot")
        return
    from sklearn.decomposition import PCA
    from matplotlib.lines import Line2D

    n_methods = len(method_names)
    n_cols = n_methods + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    class_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']
    num_classes = int(real_labels.max()) + 1

    # Subsample for visualization
    N_VIS = min(500, len(real_features))
    np.random.seed(42)
    vis_idx = np.random.choice(len(real_features), N_VIS, replace=False)
    real_vis = real_features[vis_idx]
    real_lbl_vis = real_labels[vis_idx]

    gen_vis_list = []
    gen_lbl_vis_list = []
    for i, name in enumerate(method_names):
        gf = all_gen_features[i]
        gl = all_gen_labels[i]
        n_vis_g = min(500, len(gf))
        idx_g = np.random.choice(len(gf), n_vis_g, replace=False)
        gen_vis_list.append(gf[idx_g])
        gen_lbl_vis_list.append(gl[idx_g])

    # Joint PCA + UMAP
    all_data = [real_vis]
    all_labels = [real_lbl_vis]
    for i in range(n_methods):
        n_g = len(gen_vis_list[i])
        all_data.append(gen_vis_list[i])
        all_labels.append(gen_lbl_vis_list[i])

    combined = np.vstack(all_data)
    combined_labels = np.concatenate(all_labels)

    n_pca = min(50, combined.shape[1] - 1, combined.shape[0] - 1)
    print(f"  PCA({n_pca}) + UMAP on {combined.shape[0]} samples...")
    pca = PCA(n_components=n_pca, random_state=42)
    combined_pca = pca.fit_transform(combined)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(combined_pca)

    # Split back
    n_real = N_VIS
    real_emb = embedding[:n_real]
    real_lbl_emb = combined_labels[:n_real]
    gen_emb_list = []
    gen_lbl_emb_list = []
    offset = n_real
    for i in range(n_methods):
        n_g = len(gen_vis_list[i])
        gen_emb_list.append(embedding[offset:offset + n_g])
        gen_lbl_emb_list.append(combined_labels[offset:offset + n_g])
        offset += n_g

    # Plot Real data (first column)
    ax = axes[0]
    for c in range(num_classes):
        mask = real_lbl_emb == c
        ax.scatter(real_emb[mask, 0], real_emb[mask, 1],
                   c=class_colors[c % len(class_colors)], marker='o',
                   alpha=0.3, s=25, edgecolors='none')
        cx, cy = real_emb[mask, 0].mean(), real_emb[mask, 1].mean()
        ax.scatter([cx], [cy], c=class_colors[c % len(class_colors)],
                   marker='*', s=150, edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_title('Real Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=9)

    # Plot each method
    for i, name in enumerate(method_names):
        ax = axes[i + 1]
        gen_emb = gen_emb_list[i]
        gen_lbl = gen_lbl_emb_list[i]

        # Real as background
        for c in range(num_classes):
            mask = real_lbl_emb == c
            ax.scatter(real_emb[mask, 0], real_emb[mask, 1],
                       c=class_colors[c % len(class_colors)], marker='o',
                       alpha=0.15, s=20, edgecolors='none')
            r_cx, r_cy = real_emb[mask, 0].mean(), real_emb[mask, 1].mean()

        # Generated samples
        for c in range(num_classes):
            mask = gen_lbl == c
            ax.scatter(gen_emb[mask, 0], gen_emb[mask, 1],
                       c=class_colors[c % len(class_colors)], marker='^',
                       alpha=0.4, s=30, edgecolors='none')
            g_cx, g_cy = gen_emb[mask, 0].mean(), gen_emb[mask, 1].mean()
            ax.scatter([g_cx], [g_cy], c=class_colors[c % len(class_colors)],
                       marker='P', s=100, edgecolors='black', linewidths=0.5, zorder=5)
            ax.plot([r_cx, g_cx], [r_cy, g_cy],
                    color=class_colors[c % len(class_colors)],
                    linewidth=1.5, linestyle='--', alpha=0.5)

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label='Generated'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=10, label='Real Center'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='gray',
               markersize=8, label='Gen Center'),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=9, loc='best',
                    framealpha=0.9, ncol=2)

    fig.suptitle(f'UMAP Distribution: Real vs Generated ({dataset_name})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved UMAP: {save_path} and {pdf_path}")


# ============================================================================
# Main Evaluation
# ============================================================================
def evaluate_dataset(dataset):
    """Evaluate all methods on a dataset."""
    print(f"\n{'='*80}")
    print(f"Evaluating {dataset}")
    print(f"{'='*80}")

    X, y, channels, n_samples, fs, num_classes, data_mean, data_std = load_dataset(dataset)

    # Pre-train feature extractor
    print("Pre-training feature extractor...")
    n_train = min(len(X), 2000)
    feat_extractor = pretrain_feature_extractor(
        channels, n_samples, num_classes, X[:n_train], y[:n_train], epochs=300)

    # Extract real features
    print("Extracting real features...")
    real_features_list = []
    real_probs_list = []
    batch_size = 128
    for i in range(0, len(X), batch_size):
        xb = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            logits, h = feat_extractor(xb)
            probs = F.softmax(logits, dim=1)
        real_features_list.append(h.cpu().numpy())
        real_probs_list.append(probs.cpu().numpy())
    real_features = np.vstack(real_features_list)
    real_probs = np.vstack(real_probs_list)

    # Load DDPM
    ddpm, ddpm_stats = load_ddpm(dataset, channels, n_samples, fs, num_classes)

    # Methods
    baseline_models = ['cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth']
    n_gen = 400
    n_per_class = n_gen // num_classes

    results = {}
    all_gen_features = {}
    all_gen_probs = {}
    all_gen_labels = {}

    def evaluate_method(name, gen_X, gen_y):
        """Compute all metrics for a method."""
        gen_t = torch.FloatTensor(gen_X).to(DEVICE)
        gen_feats, gen_probs_arr = [], []
        for i in range(0, len(gen_t), batch_size):
            xb = gen_t[i:i + batch_size]
            with torch.no_grad():
                logits, h = feat_extractor(xb)
                probs = F.softmax(logits, dim=1)
            gen_feats.append(h.cpu().numpy())
            gen_probs_arr.append(probs.cpu().numpy())
        gen_features = np.vstack(gen_feats)
        gen_probs_np = np.vstack(gen_probs_arr)

        is_mean, is_std = compute_inception_score(gen_probs_np, num_classes)
        fid = compute_fid(real_features, gen_features)
        sfid = compute_sfid(feat_extractor, X, gen_X)
        precision, recall = compute_precision_recall(real_features, gen_features)
        return {
            'IS': is_mean, 'IS_std': is_std, 'FID': fid, 'sFID': sfid,
            'Precision': precision, 'Recall': recall,
        }, gen_features, gen_probs_np

    # --- Gaussian Noise ---
    print("\nEvaluating gaussian_noise...")
    gen_X, gen_y = gaussian_noise_augment(X, y, n_per_class, num_classes)
    r, gf, gp = evaluate_method('gaussian_noise', gen_X, gen_y)
    results['gaussian_noise'] = r
    all_gen_features['Gaussian Noise'] = gf
    all_gen_probs['gaussian_noise'] = gp
    all_gen_labels['Gaussian Noise'] = gen_y
    print(f"  IS={r['IS']:.2f}, FID={r['FID']:.2f}, sFID={r['sFID']:.2f}, Prec={r['Precision']:.3f}, Rec={r['Recall']:.3f}")
    del gen_X
    torch.cuda.empty_cache()

    # --- SMOTE ---
    print("\nEvaluating smote...")
    gen_X, gen_y = smote_augment(X, y, n_per_class, num_classes)
    r, gf, gp = evaluate_method('smote', gen_X, gen_y)
    results['smote'] = r
    all_gen_features['SMOTE'] = gf
    all_gen_probs['smote'] = gp
    all_gen_labels['SMOTE'] = gen_y
    print(f"  IS={r['IS']:.2f}, FID={r['FID']:.2f}, sFID={r['sFID']:.2f}, Prec={r['Precision']:.3f}, Rec={r['Recall']:.3f}")
    del gen_X
    torch.cuda.empty_cache()

    # --- Baseline models ---
    for model_name in baseline_models:
        display_name = model_name.replace('_', '-').title().replace('-', ' ')
        if model_name == 'cond_ddpm':
            display_name = 'Cond-DDPM'
        elif model_name == 'diffeegbooth':
            display_name = 'DiffEEGBooth'

        print(f"\nEvaluating {model_name}...")
        model = load_baseline(model_name, dataset, channels, n_samples, fs, num_classes)
        if model is None:
            print(f"  Skipping {model_name} (no checkpoint)")
            results[model_name] = {'status': 'no_checkpoint'}
            continue

        gen_X, gen_y = generate_samples(model, model_name, n_per_class, num_classes,
                                         data_mean=data_mean, data_std=data_std)
        del model
        torch.cuda.empty_cache()

        r, gf, gp = evaluate_method(model_name, gen_X, gen_y)
        results[model_name] = r
        all_gen_features[display_name] = gf
        all_gen_probs[model_name] = gp
        all_gen_labels[display_name] = gen_y
        print(f"  IS={r['IS']:.2f}, FID={r['FID']:.2f}, sFID={r['sFID']:.2f}, Prec={r['Precision']:.3f}, Rec={r['Recall']:.3f}")
        del gen_X
        torch.cuda.empty_cache()

    # --- DDPM (Ours) ---
    if ddpm is not None:
        print("\nEvaluating DDPM (Ours)...")
        gen_X, gen_y = generate_samples(ddpm, 'ddpm', n_per_class, num_classes,
                                         data_mean=data_mean, data_std=data_std)
        del ddpm
        torch.cuda.empty_cache()

        r, gf, gp = evaluate_method('ddpm', gen_X, gen_y)
        results['ddpm'] = r
        all_gen_features['DDPM (Ours)'] = gf
        all_gen_probs['ddpm'] = gp
        all_gen_labels['DDPM (Ours)'] = gen_y
        print(f"  IS={r['IS']:.2f}, FID={r['FID']:.2f}, sFID={r['sFID']:.2f}, Prec={r['Precision']:.3f}, Rec={r['Recall']:.3f}")
        del gen_X
        torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(OUTPUT_DIR, f'{dataset}_quality_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # UMAP visualization
    method_names = list(all_gen_features.keys())
    gen_feats_list = [all_gen_features[n] for n in method_names]
    gen_labels_list = [all_gen_labels[n] for n in method_names]
    umap_path = os.path.join(OUTPUT_DIR, f'{dataset}_umap.png')
    plot_umap(real_features, gen_feats_list, method_names, y,
              gen_labels_list, umap_path, dataset_name=dataset.upper())

    # Print summary table
    print(f"\n{'='*100}")
    print(f"Quality Metrics Summary - {dataset}")
    print(f"{'='*100}")
    print(f"{'Method':<18} {'IS_up':<10} {'FID_down':<10} {'sFID_down':<10} {'Prec_up':<10} {'Recall_up':<10}")
    print("-" * 100)
    for name, r in results.items():
        if 'status' in r and r['status'] != 'success':
            print(f"{name:<18} (no checkpoint)")
        else:
            print(f"{name:<18} {r['IS']:<10.2f} {r['FID']:<10.2f} {r['sFID']:<10.2f} {r['Precision']:<10.3f} {r['Recall']:<10.3f}")
    print("=" * 100)

    return results


# ============================================================================
# Cross-dataset Summary Table
# ============================================================================
def print_cross_dataset_summary(all_results):
    """Print a combined summary table across all datasets."""
    print(f"\n{'='*120}")
    print("Cross-Dataset Quality Metrics Summary")
    print(f"{'='*120}")

    for ds_name, results in all_results.items():
        print(f"\n--- {ds_name.upper()} ---")
        print(f"{'Method':<18} {'IS_up':<10} {'FID_down':<10} {'sFID_down':<10} {'Prec_up':<10} {'Recall_up':<10}")
        print("-" * 70)
        for name, r in results.items():
            if 'status' in r:
                continue
            print(f"{name:<18} {r['IS']:<10.2f} {r['FID']:<10.2f} {r['sFID']:<10.2f} {r['Precision']:<10.3f} {r['Recall']:<10.3f}")

    # Find best methods per metric
    print(f"\n{'='*120}")
    print("Best Methods Per Metric")
    print(f"{'='*120}")
    for ds_name, results in all_results.items():
        valid = {k: v for k, v in results.items() if 'status' not in v}
        if not valid:
            continue
        best_is = max(valid.items(), key=lambda x: x[1]['IS'])
        best_fid = min(valid.items(), key=lambda x: x[1]['FID'])
        best_sfid = min(valid.items(), key=lambda x: x[1]['sFID'])
        best_prec = max(valid.items(), key=lambda x: x[1]['Precision'])
        best_rec = max(valid.items(), key=lambda x: x[1]['Recall'])
        print(f"\n{ds_name.upper()}:")
        print(f"  Best IS:    {best_is[0]} ({best_is[1]['IS']:.2f})")
        print(f"  Best FID:   {best_fid[0]} ({best_fid[1]['FID']:.2f})")
        print(f"  Best sFID:  {best_sfid[0]} ({best_sfid[1]['sFID']:.2f})")
        print(f"  Best Prec:  {best_prec[0]} ({best_prec[1]['Precision']:.3f})")
        print(f"  Best Rec:   {best_rec[0]} ({best_rec[1]['Recall']:.3f})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute generative quality metrics')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'bci2a', 'bci2b', 'physionet'],
                        help='Dataset to evaluate (default: all)')
    args = parser.parse_args()

    start_time = time.time()
    all_results = {}

    if args.dataset == 'all':
        for ds in ['bci2a', 'bci2b', 'physionet']:
            try:
                all_results[ds] = evaluate_dataset(ds)
            except Exception as e:
                print(f"Error evaluating {ds}: {e}")
                import traceback
                traceback.print_exc()
                all_results[ds] = {}
    else:
        all_results[args.dataset] = evaluate_dataset(args.dataset)

    # Print cross-dataset summary
    if len(all_results) > 1:
        print_cross_dataset_summary(all_results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
