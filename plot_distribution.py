#!/usr/bin/env python3
"""
UMAP distribution comparison for ERD_DDPM and baseline methods.

Supports: BCI2a (4-class, 22ch), BCI2b (2-class, 3ch)
Methods: Gaussian Noise, SMOTE, CVAE, WaveGAN, Cond-DDPM, BrainDiff, EEGDiff, DiffEEGBooth, DDPM (Ours)

Layout: 1 row, (1 + n_methods) columns
- Col 0: Real data only
- Col 1..N: Real (circles) + Generated (triangles) per method
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'baselines'))
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'ddpm'))

from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE
from class_discriminative import MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
from core.classifiers.eegnet import EEGNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(project_root, 'checkpoints')
OUTPUT_DIR = os.path.join(project_root, 'paper', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class EEGNetFeatureExtractor(torch.nn.Module):
    def __init__(self, channels, n_samples, num_classes):
        super().__init__()
        self.backbone = EEGNet(n_channels=channels, n_samples=n_samples, n_classes=num_classes)
        self.num_features = self.backbone.feature_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        return x

    def extract_features(self, x):
        x = x.unsqueeze(1)
        return self.backbone._forward_features(x)


def pretrain_feature_extractor(channels, n_samples, num_classes, X_train, y_train, epochs=300):
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
            logits = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_t).argmax(1) == y_t).float().mean().item()
    print(f"  Feature extractor accuracy: {acc:.4f}")
    return model


def load_dataset(dataset):
    if dataset == 'bci2a':
        data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2a')
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
        X = X * 1e6
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
        y = y - y.min()
        X, y = X[y < 4], y[y < 4]
        channels, n_samples, fs, num_classes = 22, 1000, 250, 4
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    elif dataset == 'bci2b':
        data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2b')
        X = np.load(os.path.join(data_dir, 'X.npy')).astype(np.float32)
        y = np.load(os.path.join(data_dir, 'y.npy')).astype(np.int64)
        y = y - y.min()
        X, y = X[y < 2], y[y < 2]
        channels, n_samples, fs, num_classes = 3, 1000, 250, 2
        class_names = ['Left Hand', 'Right Hand']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    data_mean = X.mean(axis=(0, 2), keepdims=True)
    data_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = ((X - data_mean) / data_std).astype(np.float32)
    print(f"{dataset}: {X.shape}, range=[{X.min():.2f}, {X.max():.2f}], classes={num_classes}")
    return X, y, channels, n_samples, fs, num_classes, data_mean, data_std, class_names


def load_ddpm(dataset, channels, n_samples, fs, num_classes):
    ckpt_map = {
        'bci2a': 'best_class_discriminative.pt',
        'bci2b': 'bci2b/trained_ddpm.pt',
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


def generate_samples(model, model_name, n_per_class, num_classes):
    gen_X, gen_y = [], []
    GEN_BATCH = 16

    with torch.no_grad():
        for c in range(num_classes):
            remaining = n_per_class
            while remaining > 0:
                batch = min(GEN_BATCH, remaining)
                y = torch.full((batch,), c, dtype=torch.long, device=DEVICE)

                if model_name == 'wavegan':
                    samples = model.generate(batch, y, DEVICE)
                elif model_name == 'cvae':
                    samples = model.generate(batch, y, DEVICE)
                elif model_name == 'ddpm':
                    samples = model.sample(batch, y, guidance_scale=1.0, device=DEVICE)
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
    gen_X = np.clip(gen_X, -50.0, 50.0).astype(np.float32)
    return gen_X, gen_y


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
                synthetic = sample.copy()
            gen_X.append(synthetic)
            gen_y.append(c)
    return np.array(gen_X), np.array(gen_y)


def plot_umap(real_features, all_gen_features, method_names, real_labels,
              all_gen_labels, save_path, dataset_name, class_names):
    from sklearn.decomposition import PCA
    import umap

    n_methods = len(method_names)
    n_cols = n_methods + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    class_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#F57F17']
    num_classes = len(class_names)

    all_feat_list = [real_features]
    all_labels_list = [real_labels]
    all_method_list = [['Real'] * len(real_labels)]
    for m, gf, gl in zip(method_names, all_gen_features, all_gen_labels):
        all_feat_list.append(gf)
        all_labels_list.append(gl)
        all_method_list.append([m] * len(gl))

    all_feat = np.concatenate(all_feat_list)
    all_labels = np.concatenate(all_labels_list)
    all_methods = np.concatenate(all_method_list)

    n_pca = min(50, all_feat.shape[1] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    all_pca = pca.fit_transform(all_feat)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    emb = reducer.fit_transform(all_pca)

    real_mask = all_methods == 'Real'
    real_emb = emb[real_mask]
    real_lbl = all_labels[real_mask]

    for c in range(num_classes):
        mask = real_lbl == c
        axes[0].scatter(real_emb[mask, 0], real_emb[mask, 1],
                        c=class_colors[c], marker='o', alpha=0.2, s=20, edgecolors='none')
        cx, cy = real_emb[mask, 0].mean(), real_emb[mask, 1].mean()
        axes[0].scatter([cx], [cy], c=class_colors[c], marker='*', s=150,
                        edgecolors='black', linewidths=0.5, zorder=5)

    axes[0].set_title('Real', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP 1', fontsize=11)
    axes[0].set_ylabel('UMAP 2', fontsize=11)
    axes[0].grid(True, alpha=0.15)
    axes[0].tick_params(labelsize=9)

    for idx, method_name in enumerate(method_names):
        ax = axes[idx + 1]

        for c in range(num_classes):
            mask = real_lbl == c
            ax.scatter(real_emb[mask, 0], real_emb[mask, 1],
                       c=class_colors[c], marker='o', alpha=0.15, s=20, edgecolors='none')
            r_cx, r_cy = real_emb[mask, 0].mean(), real_emb[mask, 1].mean()
            ax.scatter([r_cx], [r_cy], c=class_colors[c], marker='*', s=100,
                       edgecolors='black', linewidths=0.5, zorder=5)

        method_mask = all_methods == method_name
        method_emb = emb[method_mask]
        method_lbl = all_labels[method_mask]

        for c in range(num_classes):
            mask = method_lbl == c
            ax.scatter(method_emb[mask, 0], method_emb[mask, 1],
                       c=class_colors[c], marker='^', alpha=0.4, s=30, edgecolors='none')
            m_cx, m_cy = method_emb[mask, 0].mean(), method_emb[mask, 1].mean()
            ax.scatter([m_cx], [m_cy], c=class_colors[c], marker='P', s=80,
                       edgecolors='black', linewidths=0.5, zorder=5)
            r_mask = real_lbl == c
            r_cx, r_cy = real_emb[r_mask, 0].mean(), real_emb[r_mask, 1].mean()
            ax.plot([r_cx, m_cx], [r_cy, m_cy], color=class_colors[c],
                    linewidth=1.5, linestyle='--', alpha=0.5)

        ax.set_title(method_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=9)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Real'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Generated'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, label='Real Center'),
        Line2D([0], [0], marker='P', color='w', markerfacecolor='gray', markersize=8, label='Gen Center'),
    ]
    for c in range(num_classes):
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=class_colors[c],
                   markersize=8, label=class_names[c])
        )
    axes[0].legend(handles=legend_elements, fontsize=11, loc='best', framealpha=0.9, ncol=2)

    fig.suptitle(f'UMAP Distribution Comparison - {dataset_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close()


def evaluate_dataset(dataset):
    print(f"\n{'='*80}")
    print(f"Generating UMAP distribution for {dataset}")
    print(f"{'='*80}")

    X, y, channels, n_samples, fs, num_classes, data_mean, data_std, class_names = load_dataset(dataset)

    print("Pre-training feature extractor...")
    n_train = min(len(X), 2000)
    feat_extractor = pretrain_feature_extractor(
        channels, n_samples, num_classes, X[:n_train], y[:n_train], epochs=300)

    print("Extracting real features...")
    real_features_list = []
    batch_size = 128
    for i in range(0, len(X), batch_size):
        xb = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            h = feat_extractor.extract_features(xb)
        real_features_list.append(h.cpu().numpy())
    real_features = np.vstack(real_features_list)
    print(f"  Real features shape: {real_features.shape}")

    ddpm, _ = load_ddpm(dataset, channels, n_samples, fs, num_classes)

    baseline_models = ['cvae', 'wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth']
    n_gen = 400
    n_per_class = n_gen // num_classes

    all_gen_features = []
    all_gen_labels = []
    method_names = []

    print("\nEvaluating Gaussian Noise...")
    gen_X, gen_y = gaussian_noise_augment(X, y, n_per_class, num_classes)
    gen_feats = []
    for i in range(0, len(gen_X), batch_size):
        xb = torch.FloatTensor(gen_X[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            h = feat_extractor.extract_features(xb)
        gen_feats.append(h.cpu().numpy())
    all_gen_features.append(np.vstack(gen_feats))
    all_gen_labels.append(gen_y)
    method_names.append('Gaussian Noise')
    del gen_X
    print(f"  Gaussian Noise features: {all_gen_features[-1].shape}")

    print("\nEvaluating SMOTE...")
    gen_X, gen_y = smote_augment(X, y, n_per_class, num_classes)
    gen_feats = []
    for i in range(0, len(gen_X), batch_size):
        xb = torch.FloatTensor(gen_X[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            h = feat_extractor.extract_features(xb)
        gen_feats.append(h.cpu().numpy())
    all_gen_features.append(np.vstack(gen_feats))
    all_gen_labels.append(gen_y)
    method_names.append('SMOTE')
    del gen_X
    print(f"  SMOTE features: {all_gen_features[-1].shape}")

    for model_name in baseline_models:
        display_name = model_name.replace('_', '-').title().replace('-', ' ')
        if model_name == 'cond_ddpm':
            display_name = 'Cond-DDPM'
        elif model_name == 'diffeegbooth':
            display_name = 'DiffEEGBooth'

        print(f"\nEvaluating {model_name}...")
        model = load_baseline(model_name, dataset, channels, n_samples, fs, num_classes)
        if model is None:
            print(f"  Skipping {model_name}")
            continue

        print(f"  Generating samples...")
        gen_X, gen_y = generate_samples(model, model_name, n_per_class, num_classes)
        print(f"  Generated: {gen_X.shape}, range=[{gen_X.min():.2f}, {gen_X.max():.2f}]")
        del model
        torch.cuda.empty_cache()

        gen_feats = []
        for i in range(0, len(gen_X), batch_size):
            xb = torch.FloatTensor(gen_X[i:i + batch_size]).to(DEVICE)
            with torch.no_grad():
                h = feat_extractor.extract_features(xb)
            gen_feats.append(h.cpu().numpy())
        all_gen_features.append(np.vstack(gen_feats))
        all_gen_labels.append(gen_y)
        method_names.append(display_name)
        del gen_X
        torch.cuda.empty_cache()
        print(f"  {model_name} features: {all_gen_features[-1].shape}")

    if ddpm is not None:
        print("\nEvaluating DDPM (Ours)...")
        gen_X, gen_y = generate_samples(ddpm, 'ddpm', n_per_class, num_classes)
        print(f"  Generated: {gen_X.shape}, range=[{gen_X.min():.2f}, {gen_X.max():.2f}]")
        del ddpm
        torch.cuda.empty_cache()

        gen_feats = []
        for i in range(0, len(gen_X), batch_size):
            xb = torch.FloatTensor(gen_X[i:i + batch_size]).to(DEVICE)
            with torch.no_grad():
                h = feat_extractor.extract_features(xb)
            gen_feats.append(h.cpu().numpy())
        all_gen_features.append(np.vstack(gen_feats))
        all_gen_labels.append(gen_y)
        method_names.append('DDPM (Ours)')
        del gen_X
        torch.cuda.empty_cache()
        print(f"  DDPM features: {all_gen_features[-1].shape}")

    if method_names:
        save_path = os.path.join(OUTPUT_DIR, f'{dataset}_umap_distribution.png')
        print(f"\nPlotting UMAP...")
        plot_umap(real_features, all_gen_features, method_names, y,
                  all_gen_labels, save_path, dataset.upper(), class_names)


def main():
    print("START: plot_distribution.py", flush=True)
    import argparse
    import logging

    log_file = os.path.join(project_root, 'plot_distribution.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(description='Generate UMAP distribution figures')
    parser.add_argument('--dataset', choices=['bci2a', 'bci2b', 'all'], default='all',
                       help='Which dataset to generate figures for')
    args = parser.parse_args()

    try:
        if args.dataset in ('bci2a', 'all'):
            evaluate_dataset('bci2a')

        if args.dataset in ('bci2b', 'all'):
            evaluate_dataset('bci2b')

        print(f"\n{'='*80}")
        print("All UMAP figures generated successfully!")
        print(f"{'='*80}")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()