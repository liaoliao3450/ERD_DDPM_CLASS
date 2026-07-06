#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTTD-style 综合可视化与质量评估脚本

参照 DTTD 项目的风格实现：
1. PSD 功率谱密度对比图 (平均PSD, Real vs Generated, 每个类别)
2. 脑地形图 Topomap (Alpha band平均功率, RBF插值, 2行×N列布局)
3. 原始数据与生成数据 UMAP 分布图 (Real=圆, Generated=三角, 叠加显示)
4. 类内/类间距离 (EEGNet特征空间, Separation Ratio)
5. FID 距离 (EEGNet特征空间)

支持数据集: BCI2a, BCI2b, PhysioNet MI4C
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal, linalg
from scipy.interpolate import Rbf
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    from sklearn.manifold import UMAP

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Path setup
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'ddpm'))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'baselines'))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'classifiers'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from class_discriminative import MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE
from eegnet import EEGNet, EEGNetEval

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_generated_data_to_real_stats(real_data, gen_data):
    """Channel-wise statistical alignment: match gen data mean/std to real data."""
    real_data = real_data.astype(np.float32)
    gen_data = gen_data.astype(np.float32)
    real_mean = real_data.mean(axis=(0, 2), keepdims=True)
    real_std = real_data.std(axis=(0, 2), keepdims=True)
    gen_mean = gen_data.mean(axis=(0, 2), keepdims=True)
    gen_std = gen_data.std(axis=(0, 2), keepdims=True)
    eps = 1e-8
    if float(np.max(gen_std)) < eps:
        return gen_data
    gen_norm = (gen_data - gen_mean) / (gen_std + eps)
    gen_aligned = gen_norm * (real_std + eps) + real_mean
    return gen_aligned

# ============================================================================
# Matplotlib style (DTTD-style)
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# Dataset configuration (DTTD-style channel positions)
# ============================================================================
DATASET_CONFIG = {
    'bci2a': {
        'channels': 22, 'n_samples': 1000, 'fs': 250, 'num_classes': 4,
        'ckpt': 'best_class_discriminative.pt',
        'channel_names': [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ],
        # DTTD-style 10-20 normalized positions (Y+ = anterior/nose)
        'channel_positions': np.array([
            [0.0, 0.72],    # Fz
            [-0.39, 0.54],  # FC3
            [-0.17, 0.54],  # FC1
            [0.0, 0.54],    # FCz
            [0.17, 0.54],   # FC2
            [0.39, 0.54],   # FC4
            [-0.59, 0.18],  # C5
            [-0.39, 0.18],  # C3 (key)
            [-0.17, 0.18],  # C1
            [0.0, 0.18],    # Cz (key)
            [0.17, 0.18],   # C2
            [0.39, 0.18],   # C4 (key)
            [0.59, 0.18],   # C6
            [-0.39, -0.18], # CP3
            [-0.17, -0.18], # CP1
            [0.0, -0.18],   # CPz
            [0.17, -0.18],  # CP2
            [0.39, -0.18],  # CP4
            [-0.17, -0.54], # P1
            [0.0, -0.54],   # Pz
            [0.17, -0.54],  # P2
            [0.0, -0.72],   # POz
        ]),
        'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
        'key_channels': [7, 9, 11],  # C3, Cz, C4
        'key_channel_names': ['C3', 'Cz', 'C4'],
        # ERD channel mapping: class_idx -> key channel showing ERD
        'erd_channel': [11, 7, 9, 9],  # Left->C4, Right->C3, Feet->Cz, Tongue->Cz
    },
    'bci2b': {
        'channels': 3, 'n_samples': 1000, 'fs': 250, 'num_classes': 2,
        'ckpt': 'best_class_discriminative_bci2b.pt',
        'channel_names': ['C3', 'Cz', 'C4'],
        'channel_positions': np.array([
            [-0.39, 0.18],  # C3
            [0.0, 0.18],    # Cz
            [0.39, 0.18],   # C4
        ]),
        'class_names': ['Left Hand', 'Right Hand'],
        'key_channels': [0, 1, 2],
        'key_channel_names': ['C3', 'Cz', 'C4'],
        'erd_channel': [2, 0],  # Left->C4, Right->C3
    },
    'physionet': {
        'channels': 64, 'n_samples': 640, 'fs': 375, 'num_classes': 4,
        'ckpt': 'best_class_discriminative_physionet_mi4c.pt',
        'channel_names': [
            'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
            'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1', 'Oz', 'O2', 'Iz',
        ],
        'channel_positions': None,  # Generated below
        'class_names': ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
        'key_channels': [8, 10, 12],  # C3, Cz, C4
        'key_channel_names': ['C3', 'Cz', 'C4'],
        'erd_channel': [12, 8, 10, 10],  # Left->C4, Right->C3, Feet->Cz, Tongue->Cz
    },
}


def _generate_physionet_positions():
    """Generate 10-10 system positions for 64 channels (DTTD-style)."""
    names = DATASET_CONFIG['physionet']['channel_names']
    pos_map = {
        'Fp1': [-0.15, 0.85], 'Fpz': [0.0, 0.85], 'Fp2': [0.15, 0.85],
        'Af7': [-0.3, 0.75], 'Af3': [-0.1, 0.75], 'Afz': [0.0, 0.75],
        'Af4': [0.1, 0.75], 'Af8': [0.3, 0.75],
        'F7': [-0.45, 0.6], 'F5': [-0.28, 0.6], 'F3': [-0.15, 0.6],
        'F1': [-0.05, 0.6], 'Fz': [0.0, 0.6], 'F2': [0.05, 0.6],
        'F4': [0.15, 0.6], 'F6': [0.28, 0.6], 'F8': [0.45, 0.6],
        'Ft7': [-0.55, 0.36], 'Ft8': [0.55, 0.36],
        'Fc5': [-0.45, 0.36], 'Fc3': [-0.28, 0.36], 'Fc1': [-0.1, 0.36],
        'Fcz': [0.0, 0.36], 'Fc2': [0.1, 0.36], 'Fc4': [0.28, 0.36], 'Fc6': [0.45, 0.36],
        'T7': [-0.6, 0.0], 'T8': [0.6, 0.0], 'T9': [-0.65, 0.0], 'T10': [0.65, 0.0],
        'C5': [-0.45, 0.0], 'C3': [-0.28, 0.0], 'C1': [-0.1, 0.0],
        'Cz': [0.0, 0.0], 'C2': [0.1, 0.0], 'C4': [0.28, 0.0], 'C6': [0.45, 0.0],
        'Tp7': [-0.55, -0.36], 'Tp8': [0.55, -0.36],
        'Cp5': [-0.45, -0.36], 'Cp3': [-0.28, -0.36], 'Cp1': [-0.1, -0.36],
        'Cpz': [0.0, -0.36], 'Cp2': [0.1, -0.36], 'Cp4': [0.28, -0.36], 'Cp6': [0.45, -0.36],
        'P7': [-0.45, -0.6], 'P5': [-0.28, -0.6], 'P3': [-0.15, -0.6],
        'P1': [-0.05, -0.6], 'Pz': [0.0, -0.6], 'P2': [0.05, -0.6],
        'P4': [0.15, -0.6], 'P6': [0.28, -0.6], 'P8': [0.45, -0.6],
        'Po7': [-0.3, -0.75], 'Po3': [-0.1, -0.75], 'Poz': [0.0, -0.75],
        'Po4': [0.1, -0.75], 'Po8': [0.3, -0.75],
        'O1': [-0.15, -0.85], 'Oz': [0.0, -0.85], 'O2': [0.15, -0.85],
        'Iz': [0.0, -0.9],
    }
    positions = np.zeros((64, 2))
    for i, name in enumerate(names):
        key = name.capitalize()
        if key in pos_map:
            positions[i] = pos_map[key]
        else:
            angle = 2 * np.pi * i / 64
            positions[i] = [0.4 * np.cos(angle), 0.4 * np.sin(angle)]
    return positions


DATASET_CONFIG['physionet']['channel_positions'] = _generate_physionet_positions()

# Method display colors (DTTD-style)
METHOD_COLORS = {
    'Real': '#1f77b4',
    'Gaussian Noise': '#ff7f0e',
    'SMOTE': '#2ca02c',
    'CVAE': '#7B1FA2',
    'WaveGAN': '#9467bd',
    'Cond-DDPM': '#8c564b',
    'BrainDiff': '#00838F',
    'EEGDiff': '#FF8F00',
    'DiffEEGBooth': '#bcbd22',
    'DDPM (Ours)': '#E53935',
}

# Class colors (DTTD-style)
CLASS_COLORS = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']


# ============================================================================
# Data loading
# ============================================================================
def load_dataset(dataset: str):
    """Load dataset and return normalized data."""
    cfg = DATASET_CONFIG[dataset]
    data_dir = PROJECT_ROOT / 'data' / 'processed'

    if dataset == 'bci2a':
        from data_loader import load_bci2a_data
        X, y, subjects, sessions = load_bci2a_data()
    elif dataset == 'bci2b':
        from data_loader_bci2b import load_bci2b_data
        X, y, subjects, sessions = load_bci2b_data()
    elif dataset == 'physionet':
        from data_loader_physionet_mi4c import load_physionet_mi4c_data
        X, y, subjects, sessions, _ = load_physionet_mi4c_data(data_root=str(data_dir / 'PhysioNetMI4C'))

    X = X.astype(np.float32)
    y = y.astype(np.int64)
    y = y - y.min()
    mask = y < cfg['num_classes']
    X, y = X[mask], y[mask]
    if dataset != 'physionet':
        subjects, sessions = subjects[mask], sessions[mask]

    # Normalize using DDPM checkpoint stats
    ckpt_path = CKPT_DIR / cfg['ckpt']
    data_mean, data_std = None, None
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            data_mean = ckpt.get('data_mean', None)
            data_std = ckpt.get('data_std', None)

    if data_mean is not None and data_std is not None:
        if isinstance(data_mean, torch.Tensor):
            data_mean = data_mean.numpy()
            data_std = data_std.numpy()
        X = ((X - data_mean) / data_std).astype(np.float32)
    else:
        dm = X.mean(axis=(0, 2), keepdims=True)
        ds = np.maximum(X.std(axis=(0, 2), keepdims=True), 1e-6)
        X = ((X - dm) / ds).astype(np.float32)
    X = np.clip(X, -5.0, 5.0)

    print(f"[{dataset}] {X.shape}, range=[{X.min():.2f}, {X.max():.2f}], classes={cfg['num_classes']}")
    return X, y, subjects, sessions


# ============================================================================
# Model loading and sample generation
# ============================================================================
def load_ddpm_model(dataset: str):
    cfg = DATASET_CONFIG[dataset]
    ckpt_path = CKPT_DIR / cfg['ckpt']
    if not ckpt_path.exists():
        print(f"  No DDPM checkpoint: {ckpt_path}")
        return None

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    eps_model = MultiScaleCondUNet(channels=cfg['channels'], num_classes=cfg['num_classes']).to(DEVICE)
    classifier = EEGClassifier(channels=cfg['channels'], n_samples=cfg['n_samples'],
                               num_classes=cfg['num_classes']).to(DEVICE)

    target_psd = ckpt.get("target_psd", torch.zeros(cfg['n_samples'] // 2 + 1, device=DEVICE)).to(DEVICE)
    target_lat = ckpt.get("target_laterality", torch.zeros(cfg['num_classes'], device=DEVICE)).to(DEVICE)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_lat,
        n_timesteps=1000, channels=cfg['channels'], n_samples=cfg['n_samples'], fs=cfg['fs'],
    ).to(DEVICE)

    if "model_state_dict" in ckpt:
        ddpm.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        ddpm.load_state_dict(ckpt, strict=False)
    ddpm.eval()
    return ddpm


def load_baseline_model(model_name: str, dataset: str):
    cfg = DATASET_CONFIG[dataset]
    ckpt_path = CKPT_DIR / 'baselines' / f'{model_name}_{dataset}.pt'
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    ch, ns, fs, nc = cfg['channels'], cfg['n_samples'], cfg['fs'], cfg['num_classes']

    if model_name == 'cvae':
        model = CVAE(channels=ch, latent_dim=64, out_length=ns, num_classes=nc)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'wavegan':
        model = WaveGAN(channels=ch, out_length=ns, num_classes=nc)
        model.generator.load_state_dict(ckpt['generator'])
    elif model_name == 'cond_ddpm':
        model = CondDDPM(channels=ch, n_samples=ns, num_classes=nc)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'braindiff':
        model = BrainDiff(channels=ch, n_samples=ns, num_classes=nc)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'eegdiff':
        model = EEGDiff(channels=ch, n_samples=ns, num_classes=nc)
        model.load_state_dict(ckpt['model_state_dict'])
    elif model_name == 'diffeegbooth':
        model = DiffEEGBooth(channels=ch, n_samples=ns, num_classes=nc, fs=fs)
        sd = ckpt['model_state_dict']
        if 'target_laterality' in sd and sd['target_laterality'].shape[0] != nc:
            sd['target_laterality'] = torch.zeros(nc)
        model.load_state_dict(sd, strict=True)
    else:
        return None

    return model.to(DEVICE).eval()


def generate_samples(model, model_name: str, n_per_class: int, num_classes: int):
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
                    samples = model.sample_ddim(batch, y, steps=50, guidance_scale=3.0, device=str(DEVICE))
                elif model_name in ('cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth'):
                    samples = model.sample_ddim(batch, y, steps=50, device=DEVICE)
                else:
                    continue

                # No clamp: real data range is [-14.5, 13.3], clamping to [-5,5]
                # destroys distribution tails and inflates FID
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch)
                remaining -= batch
                del samples, y
                torch.cuda.empty_cache()

    return np.concatenate(gen_X), np.array(gen_y)


def gaussian_noise_augment(X_train, y_train, n_per_class, num_classes):
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
# 1. PSD Power Spectral Density Comparison (Average, DTTD-style)
# ============================================================================
def plot_psd_comparison(real_X, real_y, gen_dict, cfg, dataset_name, save_dir):
    """
    Plot average PSD comparison between real and generated data for each class.
    DTTD-style: 2x2 layout for 4-class, key channel per class (ERD channel).
    Each subplot shows average PSD (solid=Real, dashed=Generated).
    """
    num_classes = cfg['num_classes']
    fs = cfg['fs']
    class_names = cfg['class_names']
    erd_channels = cfg['erd_channel']
    ch_names = cfg['channel_names']

    methods = list(gen_dict.keys())

    # Layout: num_classes rows x 1 col (one subplot per class, all methods overlaid)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))
    if num_classes == 1:
        axes = [axes]

    for c in range(num_classes):
        ax = axes[c]
        ch_idx = erd_channels[c]
        ch_name = ch_names[ch_idx]

        # Real average PSD
        real_class = real_X[real_y == c]
        freqs, psd_all = signal.welch(real_class[:, ch_idx, :], fs=fs,
                                      nperseg=min(256, real_class.shape[2]), axis=-1)
        psd_mean = psd_all.mean(axis=0)
        psd_std = psd_all.std(axis=0) / np.sqrt(len(psd_all))
        freq_mask = freqs <= 50

        ax.semilogy(freqs[freq_mask], psd_mean[freq_mask], '-',
                    color=METHOD_COLORS['Real'], linewidth=2.5, label='Real', alpha=0.9)
        ax.fill_between(freqs[freq_mask],
                        (psd_mean - psd_std)[freq_mask],
                        (psd_mean + psd_std)[freq_mask],
                        color=METHOD_COLORS['Real'], alpha=0.15)

        # Generated average PSD for each method
        for mname in methods:
            gen_X, gen_y = gen_dict[mname]
            gen_class = gen_X[gen_y == c]
            if len(gen_class) == 0:
                continue
            _, psd_gen_all = signal.welch(gen_class[:, ch_idx, :], fs=fs,
                                          nperseg=min(256, gen_class.shape[2]), axis=-1)
            psd_gen_mean = psd_gen_all.mean(axis=0)
            color = METHOD_COLORS.get(mname, '#888888')
            ax.semilogy(freqs[freq_mask], psd_gen_mean[freq_mask], '--',
                        color=color, linewidth=1.8, label=mname, alpha=0.7)

        # Mark frequency bands
        ax.axvspan(8, 13, alpha=0.1, color='green', label='Alpha (8-13 Hz)')
        ax.axvspan(13, 30, alpha=0.08, color='orange', label='Beta (13-30 Hz)')

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (V²/Hz)', fontsize=11)
        ax.set_title(f'{class_names[c]} MI - Channel {ch_name}', fontsize=13, fontweight='bold')
        ax.set_xlim([0, 50])
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(f'Average PSD Comparison: Real vs Generated ({dataset_name})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{dataset_name}_psd_comparison.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved PSD: {save_dir}/{dataset_name}_psd_comparison.png")


# ============================================================================
# 2. Topographic Map (Brain Topomap, DTTD-style)
# ============================================================================
def plot_topomap_comparison(real_X, real_y, gen_dict, cfg, dataset_name, save_dir):
    """
    DTTD-style topographic maps: 2 rows × num_classes cols.
    Row 0: Real data alpha band power
    Row 1: Best generated method (DDPM) alpha band power
    Each class has independent color scale.
    Uses RBF multiquadric interpolation.
    """
    num_classes = cfg['num_classes']
    fs = cfg['fs']
    ch_pos = cfg['channel_positions']
    class_names = cfg['class_names']
    n_channels = cfg['channels']

    # Use DDPM (Ours) if available, otherwise first method
    best_method = None
    for m in ['DDPM (Ours)'] + list(gen_dict.keys()):
        if m in gen_dict:
            best_method = m
            break
    if best_method is None:
        print("  No generated data for topomap")
        return

    gen_X, gen_y = gen_dict[best_method]

    # Alpha band: 8-13 Hz
    alpha_low, alpha_high = 8, 13

    # Create interpolation grid
    grid_res = 200
    xi = np.linspace(-1, 1, grid_res)
    yi = np.linspace(-1, 1, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # Head radius
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius

    # Figure: 2 rows (Real, Generated) x num_classes cols
    fig, axes = plt.subplots(2, num_classes, figsize=(4.5 * num_classes, 9))
    if num_classes == 1:
        axes = axes[:, np.newaxis]

    for c in range(num_classes):
        # Compute alpha band power (average across samples)
        real_class = real_X[real_y == c]
        n_sub = min(200, len(real_class))
        real_sub = real_class[:n_sub]

        real_powers = np.zeros(n_channels)
        for ch in range(n_channels):
            f, psd = signal.welch(real_sub[:, ch, :], fs=fs, nperseg=min(256, real_sub.shape[2]))
            idx_band = (f >= alpha_low) & (f <= alpha_high)
            real_powers[ch] = psd[:, idx_band].mean()

        gen_class = gen_X[gen_y == c]
        n_sub_g = min(200, len(gen_class))
        gen_sub = gen_class[:n_sub_g]

        gen_powers = np.zeros(n_channels)
        for ch in range(n_channels):
            f, psd = signal.welch(gen_sub[:, ch, :], fs=fs, nperseg=min(256, gen_sub.shape[2]))
            idx_band = (f >= alpha_low) & (f <= alpha_high)
            gen_powers[ch] = psd[:, idx_band].mean()

        # Per-class color range
        vmin = min(real_powers.min(), gen_powers.min())
        vmax = max(real_powers.max(), gen_powers.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        # Normalize
        real_norm = (real_powers - vmin) / (vmax - vmin)
        gen_norm = (gen_powers - vmin) / (vmax - vmin)

        for row_idx, (data, row_label) in enumerate(
                [(real_norm, 'Real'), (gen_norm, best_method)]):
            ax = axes[row_idx, c]

            # RBF interpolation (DTTD-style: multiquadric)
            rbf = Rbf(ch_pos[:, 0], ch_pos[:, 1], data,
                      function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan

            # Contourf (DTTD-style: RdBu_r)
            levels = np.linspace(0, 1, 50)
            ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both')

            # Head outline
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)

            # Nose
            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)

            # Ears
            ear_l = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_r = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_l)
            ax.add_patch(ear_r)

            # Channel positions
            ax.scatter(ch_pos[:, 0], ch_pos[:, 1], c='k', s=8, zorder=5)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_aspect('equal')
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(f'{class_names[c]}', fontsize=13, fontweight='bold')
            if c == 0:
                ax.set_ylabel(row_label, fontsize=14, fontweight='bold', rotation=0, labelpad=40)

    fig.suptitle(f'Alpha Band (8-13 Hz) Topographic Maps ({dataset_name})\n'
                 f'Blue=Low Power (ERD), Red=High Power',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.05, wspace=0.08, hspace=0.15)

    for fmt in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{dataset_name}_topomap_comparison.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Topomap: {save_dir}/{dataset_name}_topomap_comparison.png")


# ============================================================================
# 3. UMAP Distribution Comparison (DTTD-style)
# ============================================================================
def train_eegnet_classifier(X_train, y_train, cfg, epochs=80):
    """Train EEGNet classifier on real data for feature extraction (DTTD-style)."""
    clf = EEGNet(
        n_channels=cfg['channels'], n_samples=cfg['n_samples'],
        n_classes=cfg['num_classes'],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Normalize
    ch_mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    ch_std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    normed = ((X_train - ch_mean) / ch_std).astype(np.float32)

    n = len(normed)
    clf.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            data_t = torch.FloatTensor(normed[idx]).unsqueeze(1).to(DEVICE)
            labels_t = torch.LongTensor(y_train[idx]).to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(clf(data_t), labels_t)
            loss.backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        all_pred = []
        for start in range(0, n, 256):
            data_t = torch.FloatTensor(normed[start:start + 256]).unsqueeze(1).to(DEVICE)
            all_pred.append(clf(data_t).argmax(dim=1).cpu().numpy())
    acc = (np.concatenate(all_pred) == y_train).mean()
    print(f"  EEGNet accuracy on real data: {acc:.4f}")

    return clf, ch_mean, ch_std


def extract_eegnet_features(clf, data, ch_mean, ch_std, batch_size=128):
    """Extract EEGNet intermediate features (DTTD-style, from flatten layer)."""
    feats = []
    for i in range(0, len(data), batch_size):
        batch = ((data[i:i + batch_size] - ch_mean) / ch_std).astype(np.float32)
        data_t = torch.FloatTensor(batch).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            x = clf._forward_features(data_t)
        feats.append(x.cpu().numpy())
    return np.concatenate(feats)


def plot_umap_comparison(real_features, real_labels, gen_features_dict, gen_labels_dict,
                         method_names, dataset_name, save_dir, num_classes):
    """
    DTTD-style joint UMAP distribution comparison.
    
    ALL methods + real data are embedded together via joint PCA(50) + UMAP.
    Each panel shows real (faded) + one method (prominent) with centroid lines.
    Layout: 2 rows, auto columns
    """
    n_methods = len(method_names)
    n_total = n_methods + 1  # real + all methods

    # 2-row layout
    n_cols = (n_total + 1) // 2  # ceil
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows))
    axes_flat = axes.flatten()

    # Subsample for visualization
    N_VIS = min(500, len(real_features))
    np.random.seed(42)
    vis_idx = np.random.choice(len(real_features), N_VIS, replace=False)
    real_vis = real_features[vis_idx]
    real_lbl_vis = real_labels[vis_idx]

    # Prepare generated data
    gen_vis_list, gen_lbl_vis_list = [], []
    for name in method_names:
        gf = gen_features_dict[name]
        gl = gen_labels_dict[name]
        n_vis_g = min(500, len(gf))
        idx_g = np.random.choice(len(gf), n_vis_g, replace=False)
        gen_vis_list.append(gf[idx_g])
        gen_lbl_vis_list.append(gl[idx_g])

    # ---- Joint PCA + UMAP (DTTD-style) ----
    all_feat = [real_vis]
    all_labels = [real_lbl_vis]
    all_method = [np.array(['Real'] * N_VIS)]
    for i in range(n_methods):
        all_feat.append(gen_vis_list[i])
        all_labels.append(gen_lbl_vis_list[i])
        all_method.append(np.array([method_names[i]] * len(gen_vis_list[i])))

    combined = np.concatenate(all_feat)
    combined_labels = np.concatenate(all_labels)
    combined_method = np.concatenate(all_method)

    n_pca = min(50, combined.shape[1] - 1, combined.shape[0] - 1)
    print(f"  Joint PCA({n_pca}) + UMAP on {combined.shape[0]} samples...")
    pca = PCA(n_components=n_pca, random_state=42)
    combined_pca = pca.fit_transform(combined)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    emb = reducer.fit_transform(combined_pca)

    # Split embedding by method
    real_emb = emb[combined_method == 'Real']
    real_lbl_emb = combined_labels[combined_method == 'Real']

    # Plot Real (first panel)
    ax = axes_flat[0]
    for c in range(num_classes):
        mask = real_lbl_emb == c
        ax.scatter(real_emb[mask, 0], real_emb[mask, 1],
                   c=CLASS_COLORS[c % len(CLASS_COLORS)], marker='o',
                   alpha=0.3, s=25, edgecolors='none')
        cx, cy = real_emb[mask, 0].mean(), real_emb[mask, 1].mean()
        ax.scatter([cx], [cy], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                   marker='*', s=150, edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_title('Real Data', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15)

    # Plot each method
    for i, name in enumerate(method_names):
        ax = axes_flat[i + 1]
        method_emb = emb[combined_method == name]
        method_lbl = combined_labels[combined_method == name]

        for c in range(num_classes):
            # Real (faded background)
            mask_r = real_lbl_emb == c
            ax.scatter(real_emb[mask_r, 0], real_emb[mask_r, 1],
                       c=CLASS_COLORS[c % len(CLASS_COLORS)], marker='o',
                       alpha=0.15, s=20, edgecolors='none')
            r_cx = real_emb[mask_r, 0].mean()
            r_cy = real_emb[mask_r, 1].mean()

            # Generated (prominent)
            mask_g = method_lbl == c
            ax.scatter(method_emb[mask_g, 0], method_emb[mask_g, 1],
                       c=CLASS_COLORS[c % len(CLASS_COLORS)], marker='^',
                       alpha=0.4, s=30, edgecolors='none')
            g_cx = method_emb[mask_g, 0].mean()
            g_cy = method_emb[mask_g, 1].mean()

            # Centroid markers
            ax.scatter([g_cx], [g_cy], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                       marker='P', s=100, edgecolors='black', linewidths=0.5, zorder=5)

            # Dashed line connecting centroids
            ax.plot([r_cx, g_cx], [r_cy, g_cy],
                    color=CLASS_COLORS[c % len(CLASS_COLORS)],
                    linewidth=1.5, linestyle='--', alpha=0.5)

        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=13)
        ax.set_ylabel('UMAP 2', fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.15)

    # Hide unused axes
    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Generated'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, label='Real Center'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', markersize=8, label='Gen Center'),
    ]
    axes_flat[n_total - 1].legend(handles=legend_elements, fontsize=11, loc='best', framealpha=0.9, ncol=2)

    fig.suptitle(f'UMAP Distribution: Real vs Generated ({dataset_name})',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{dataset_name}_umap_comparison.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved UMAP: {save_dir}/{dataset_name}_umap_comparison.png")


# ============================================================================
# 4. Intra/Inter-class Distance (DTTD-style: EEGNet features)
# ============================================================================
def compute_separation_ratio(features, labels):
    """DTTD-style separation ratio computation."""
    unique_labels = np.unique(labels)
    class_centers = []
    for lbl in unique_labels:
        mask = labels == lbl
        class_centers.append(features[mask].mean(axis=0))
    class_centers = np.array(class_centers)

    n_classes = len(unique_labels)
    inter_dists = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            inter_dists.append(np.linalg.norm(class_centers[i] - class_centers[j]))
    inter_class_distance = np.mean(inter_dists)

    intra_dists = []
    for lbl in unique_labels:
        mask = labels == lbl
        cls_feats = features[mask]
        center = cls_feats.mean(axis=0)
        dists = np.linalg.norm(cls_feats - center, axis=1)
        intra_dists.append(dists.mean())
    intra_class_distance = np.mean(intra_dists)

    separation_ratio = inter_class_distance / (intra_class_distance + 1e-10)
    return separation_ratio, inter_class_distance, intra_class_distance


def compute_mmd(x, y, sigma=None):
    """DTTD-style MMD computation with median heuristic for bandwidth."""
    from scipy.spatial.distance import cdist
    xx = cdist(x, x, 'sqeuclidean')
    yy = cdist(y, y, 'sqeuclidean')
    xy = cdist(x, y, 'sqeuclidean')

    # Median heuristic: sigma = median of pairwise distances
    if sigma is None:
        all_dists = np.concatenate([xx[np.triu_indices_from(xx, k=1)],
                                    yy[np.triu_indices_from(yy, k=1)],
                                    xy.ravel()])
        sigma = max(np.median(np.sqrt(all_dists[all_dists > 0])), 1e-6)
        print(f"    MMD sigma (median heuristic): {sigma:.4f}")

    c2 = -1.0 / (2 * sigma ** 2)
    k_xx = np.exp(c2 * xx)
    k_yy = np.exp(c2 * yy)
    k_xy = np.exp(c2 * xy)

    m = x.shape[0]
    n = y.shape[0]
    mmd = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1) + 1e-10) + \
          (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1) + 1e-10) - \
          2 * k_xy.sum() / (m * n + 1e-10)
    return max(mmd, 0.0)


def plot_distance_comparison(real_features, real_labels, gen_features_dict, gen_labels_dict,
                             method_names, dataset_name, save_dir, num_classes):
    """DTTD-style distance metrics computation and bar chart."""
    class_names = DATASET_CONFIG[dataset_name.lower()]['class_names']

    # Compute for real
    real_sep, real_inter, real_intra = compute_separation_ratio(real_features, real_labels)

    # Compute for each method (combined real + generated)
    results = {'Real': {
        'separation_ratio': float(real_sep),
        'inter_class_distance': float(real_inter),
        'intra_class_distance': float(real_intra),
    }}

    for name in method_names:
        gen_feats = gen_features_dict[name]
        gen_lbls = gen_labels_dict[name]
        combined_feats = np.vstack([real_features, gen_feats])
        combined_lbls = np.concatenate([real_labels, gen_lbls])
        sep, inter, intra = compute_separation_ratio(combined_feats, combined_lbls)
        results[name] = {
            'separation_ratio': float(sep),
            'inter_class_distance': float(inter),
            'intra_class_distance': float(intra),
        }

    # Save JSON
    json_path = os.path.join(save_dir, f'{dataset_name}_distance_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved Distance metrics: {json_path}")

    # Bar chart: Separation Ratio
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = ['Real'] + method_names
    ratios = [results[n]['separation_ratio'] for n in names]
    colors = [METHOD_COLORS.get(n, '#888888') for n in names]

    axes[0].bar(range(len(names)), ratios, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Separation Ratio (Inter/Intra)', fontsize=11)
    axes[0].set_title('Separation Ratio', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Grouped bar: Intra-class distance per class
    x = np.arange(num_classes)
    width = 0.8 / (len(names))
    for i, name in enumerate(names):
        intra_per_class = []
        for c in range(num_classes):
            if name == 'Real':
                mask = real_labels == c
                feats = real_features[mask]
            else:
                gen_lbls = gen_labels_dict[name]
                gen_feats = gen_features_dict[name]
                combined = np.vstack([real_features, gen_feats])
                combined_lbls = np.concatenate([real_labels, gen_lbls])
                mask = combined_lbls == c
                feats = combined[mask]
            center = feats.mean(axis=0)
            dist = np.linalg.norm(feats - center, axis=1).mean()
            intra_per_class.append(dist)
        axes[1].bar(x + i * width, intra_per_class, width, label=name, alpha=0.8,
                    color=colors[i])

    axes[1].set_xticks(x + width * len(names) / 2)
    axes[1].set_xticklabels(class_names, fontsize=10)
    axes[1].set_ylabel('Intra-class Distance', fontsize=11)
    axes[1].set_title('Intra-class Distance per Class', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Distance Metrics: Real vs Augmented ({dataset_name})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{dataset_name}_distance_comparison.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Distance: {save_dir}/{dataset_name}_distance_comparison.png")


# ============================================================================
# 5. FID Computation (DTTD-style: EEGNet features)
# ============================================================================
def compute_fid(real_features, gen_features):
    """DTTD-style FID computation."""
    mu_r = real_features.mean(axis=0)
    mu_g = gen_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return max(float(fid), 0.0)


def compute_all_fid(real_features, gen_features_dict, method_names, dataset_name, save_dir):
    """Compute FID for each method and plot bar chart."""
    fid_results = {}
    for name in method_names:
        gen_feats = gen_features_dict[name]
        fid = compute_fid(real_features, gen_feats)
        fid_results[name] = fid
        print(f"  FID({name}): {fid:.2f}")

    # Save JSON
    json_path = os.path.join(save_dir, f'{dataset_name}_fid_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(fid_results, f, indent=2)
    print(f"  Saved FID metrics: {json_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(fid_results.keys())
    fids = [fid_results[n] for n in names]
    colors = [METHOD_COLORS.get(n, '#888888') for n in names]
    bars = ax.bar(range(len(names)), fids, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('FID (lower is better)', fontsize=12)
    ax.set_title(f'FID Comparison ({dataset_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels on bars
    for bar, val in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    for fmt in ['png', 'pdf']:
        path = os.path.join(save_dir, f'{dataset_name}_fid_comparison.{fmt}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved FID plot: {save_dir}/{dataset_name}_fid_comparison.png")

    return fid_results


# ============================================================================
# Main pipeline
# ============================================================================
def run_visualization(dataset: str, n_gen_per_class: int = 100):
    """Run all visualizations for a dataset."""
    cfg = DATASET_CONFIG[dataset]
    save_dir = str(OUTPUT_DIR)
    dataset_name = dataset.upper()

    print(f"\n{'=' * 80}")
    print(f"DTTD-style Visualization & Quality Assessment: {dataset_name}")
    print(f"{'=' * 80}")

    # Load data
    X, y, subjects, sessions = load_dataset(dataset)

    # Subsample real data for efficiency
    N_REAL = min(2000, len(X))
    np.random.seed(42)
    real_idx = np.random.choice(len(X), N_REAL, replace=False)
    X_real = X[real_idx]
    y_real = y[real_idx]

    # Cache file for generated data
    cache_dir = Path(save_dir) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{dataset_name}_gen_cache.npz'

    if cache_file.exists():
        print(f"\nLoading cached generated data from {cache_file}...")
        cache_data = np.load(str(cache_file), allow_pickle=True)
        gen_dict = {}
        method_names = list(cache_data['method_names'])
        for name in method_names:
            gen_dict[name] = (cache_data[f'{name}_X'], cache_data[f'{name}_y'])
        print(f"  Loaded {len(method_names)} methods from cache")
    else:
        # Load DDPM
        print("\nLoading DDPM model...")
        ddpm = load_ddpm_model(dataset)

        # Generate samples from all methods
        print(f"\nGenerating {n_gen_per_class} samples per class...")
        gen_dict = {}
        n_per_class = n_gen_per_class

        # Gaussian Noise
        print("  Generating Gaussian Noise...")
        gen_X, gen_y = gaussian_noise_augment(X_real, y_real, n_per_class, cfg['num_classes'])
        gen_dict['Gaussian Noise'] = (gen_X, gen_y)

        # SMOTE
        print("  Generating SMOTE...")
        gen_X, gen_y = smote_augment(X_real, y_real, n_per_class, cfg['num_classes'])
        gen_dict['SMOTE'] = (gen_X, gen_y)

        # Baseline models
        baseline_names = {
            'cvae': 'CVAE', 'wavegan': 'WaveGAN', 'cond_ddpm': 'Cond-DDPM',
            'braindiff': 'BrainDiff', 'eegdiff': 'EEGDiff', 'diffeegbooth': 'DiffEEGBooth',
        }
        for model_key, display_name in baseline_names.items():
            print(f"  Loading {display_name}...")
            model = load_baseline_model(model_key, dataset)
            if model is not None:
                gen_X, gen_y = generate_samples(model, model_key, n_per_class, cfg['num_classes'])
                gen_dict[display_name] = (gen_X, gen_y)
                del model
                torch.cuda.empty_cache()
            else:
                print(f"    No checkpoint for {display_name}")

        # DDPM (Ours)
        if ddpm is not None:
            print("  Generating DDPM (Ours)...")
            gen_X, gen_y = generate_samples(ddpm, 'ddpm', n_per_class, cfg['num_classes'])
            gen_dict['DDPM (Ours)'] = (gen_X, gen_y)

        # Save cache
        method_names = list(gen_dict.keys())
        save_dict = {'method_names': np.array(method_names)}
        for name in method_names:
            save_dict[f'{name}_X'] = gen_dict[name][0]
            save_dict[f'{name}_y'] = gen_dict[name][1]
        np.savez(str(cache_file), **save_dict)
        print(f"  Cached generated data to {cache_file}")

    method_names = list(gen_dict.keys())
    print(f"\nMethods to visualize: {method_names}")

    # ---- UMAP Distribution only ----
    print("\nUMAP Distribution...")

    # DTTD-style: per-channel normalization using real data stats
    # Both real and generated data use the SAME ch_mean/ch_std from real data
    print("  Normalizing data (DTTD-style per-channel z-score)...")
    ch_mean = X_real.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    ch_std = X_real.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8

    print("  Training EEGNet classifier on normalized real data...")
    clf, clf_ch_mean, clf_ch_std = train_eegnet_classifier(
        ((X_real - ch_mean) / ch_std).astype(np.float32), y_real, cfg, epochs=200)

    print("  Extracting real features...")
    real_features = extract_eegnet_features(clf, ((X_real - ch_mean) / ch_std).astype(np.float32), clf_ch_mean, clf_ch_std)

    gen_features_dict = {}
    gen_labels_dict = {}
    for name in method_names:
        gen_X, gen_y = gen_dict[name]
        # Use same per-channel normalization as real data (DTTD-style)
        gen_X_normed = ((gen_X - ch_mean) / ch_std).astype(np.float32)
        print(f"  Extracting {name} features...")
        gen_features_dict[name] = extract_eegnet_features(clf, gen_X_normed, clf_ch_mean, clf_ch_std)
        gen_labels_dict[name] = gen_y

    plot_umap_comparison(real_features, y_real, gen_features_dict, gen_labels_dict,
                         method_names, dataset_name, save_dir, cfg['num_classes'])

    # Cleanup
    del ddpm, clf
    torch.cuda.empty_cache()

    print(f"\nAll visualizations saved to: {save_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DTTD-style Visualization & Quality Assessment')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'bci2a', 'bci2b', 'physionet'])
    parser.add_argument('--n_gen', type=int, default=100,
                        help='Number of samples to generate per class per method')
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in ['bci2a', 'bci2b', 'physionet']:
            run_visualization(ds, n_gen_per_class=args.n_gen)
    else:
        run_visualization(args.dataset, n_gen_per_class=args.n_gen)
