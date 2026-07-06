#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Baseline Training Script for Comparative Experiments

Supports multiple datasets: BCI2a, PhysioNet, HighGamma
Trains comparison methods:
1. WaveGAN - GAN-based baseline
2. Cond-DDPM - Standard conditional DDPM (no ERD/spectral)
3. BrainDiff - RL-guided diffusion
4. EEGDiff - Transformer-based diffusion
5. DiffEEGBooth - 3D structured EEG + ERD/ERS constraints
6. CVAE - Conditional Variational Autoencoder

Note: Gaussian Noise and SMOTE do not require training.

Usage:
    python train_baselines_bci2a.py --dataset bci2a --model wavegan
    python train_baselines_bci2a.py --dataset physionet --model cvae
    python train_baselines_bci2a.py --dataset highgamma --model wavegan
    python train_baselines_bci2a.py --dataset bci2a --model all
"""
import os
import sys
import io
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Output encoding
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'ddpm'))
sys.path.insert(0, os.path.join(project_root, 'core', 'models', 'baselines'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

from comparison_models import (
    create_baseline_model, WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
)
from cvae_gaussian import CVAE
from class_discriminative import EEGClassifier, pretrain_classifier


def load_bci2a_data():
    """Load BCI2a data directly from processed files."""
    data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2a')
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Data not found: {X_path}")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    # Normalize labels to 0..num_classes-1
    y = y - y.min()
    mask = y < 4
    X, y = X[mask], y[mask]

    # Standardize per-channel
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

    # Create subject and session IDs (9 subjects, 2 sessions)
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


def load_physionet_data():
    """Load PhysioNet MI4C data."""
    data_dir = os.path.join(project_root, 'data', 'processed', 'PhysioNetMI4C')
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"PhysioNet data not found: {X_path}")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    y = y - y.min()

    # Standardize
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

    subjects = np.arange(len(X))

    print(f"PhysioNet data loaded: {X.shape}, classes: {np.bincount(y)}")
    return X, y, subjects, np.zeros(len(X), dtype=int)


def load_high_gamma_data():
    """Load High Gamma dataset."""
    data_dir = os.path.join(project_root, 'data', 'processed', 'HighGamma')
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"High Gamma data not found: {X_path}")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    y = y - y.min()

    # Standardize
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

    subjects = np.arange(len(X))
    sessions = np.zeros(len(X), dtype=int)

    print(f"High Gamma data loaded: {X.shape}, classes: {np.bincount(y)}")
    return X, y, subjects, sessions


def load_bci2b_data():
    """Load BCI2b data (3 channels, 2 classes)."""
    data_dir = os.path.join(project_root, 'data', 'processed', 'BCI2b')
    X_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"BCI2b data not found: {X_path}")

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    y = y - y.min()

    # Standardize
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

    # BCI2b: 9 subjects, 2 sessions (T=0, E=1)
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

    print(f"BCI2b data loaded: {X.shape}, classes: {np.bincount(y)}, subjects: {len(np.unique(subjects))}")
    return X, y, subjects, sessions


# Dataset configurations
DATASET_CONFIG = {
    'bci2a': {
        'channels': 22,
        'n_samples': 1000,
        'fs': 250,
        'num_classes': 4,
        'load_fn': load_bci2a_data,
    },
    'physionet': {
        'channels': 64,
        'n_samples': 640,
        'fs': 160,
        'num_classes': 4,
        'load_fn': load_physionet_data,
    },
    'highgamma': {
        'channels': 128,
        'n_samples': 1000,
        'fs': 512,
        'num_classes': 4,
        'load_fn': load_high_gamma_data,
    },
    'bci2b': {
        'channels': 3,
        'n_samples': 1000,
        'fs': 250,
        'num_classes': 2,
        'load_fn': load_bci2b_data,
    },
}

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints', 'baselines')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# BCI2a parameters
CHANNELS = 22
N_SAMPLES = 1000
FS = 250
NUM_CLASSES = 4

# Training parameters
EPOCHS = 1000
BATCH_SIZE = 32
LR = 1e-4
N_TIMESTEPS = 1000


# ============================================================================
# Training Functions
# ============================================================================

def train_wavegan(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=1e-4, dataset_name='bci2a', start_epoch=0):
    """Train WaveGAN with WGAN-GP (stabilized)"""
    print(f"\n{'='*60}")
    print(f"Training WaveGAN (WGAN-GP) (starting from epoch {start_epoch})")
    print(f"{'='*60}")

    X = torch.FloatTensor(X_train).to(DEVICE)
    y = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = model.generator.to(DEVICE)
    D = model.discriminator.to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.9))

    n_critic = 5
    lambda_gp = 10

    best_g_loss = float('inf')

    for epoch in range(1 + start_epoch, epochs + 1):
        d_losses, g_losses, d_real_vals, d_fake_vals, gp_vals = [], [], [], [], []

        for i, (xb, yb) in enumerate(loader):
            bs = xb.size(0)

            # Train Discriminator
            for _ in range(n_critic):
                z = torch.randn(bs, model.z_dim, device=DEVICE)
                fake = G(z, yb).detach()

                d_real = D(xb, yb)
                d_fake = D(fake, yb)

                # Gradient penalty
                alpha_gp = torch.rand(bs, 1, 1, device=DEVICE)
                interp = (alpha_gp * xb + (1 - alpha_gp) * fake).requires_grad_(True)
                d_interp = D(interp, yb)
                grad = torch.autograd.grad(
                    d_interp.sum(), interp, create_graph=True
                )[0]
                gp = ((grad.norm(dim=1) - 1) ** 2).mean()

                d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # Train Generator
            z = torch.randn(bs, model.z_dim, device=DEVICE)
            fake = G(z, yb)
            d_fake_g = D(fake, yb)
            g_loss = -d_fake_g.mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            d_real_vals.append(d_real.mean().item())
            d_fake_vals.append(d_fake_g.mean().item())
            gp_vals.append(gp.item())

        avg_d = np.mean(d_losses)
        avg_g = np.mean(g_losses)
        avg_d_real = np.mean(d_real_vals)
        avg_d_fake = np.mean(d_fake_vals)
        avg_gp = np.mean(gp_vals)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs}: D_loss={avg_d:.4f}, G_loss={avg_g:.4f}, "
                  f"D_real={avg_d_real:.4f}, D_fake={avg_d_fake:.4f}, GP={avg_gp:.4f}")

        # Save best
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            save_path = os.path.join(CHECKPOINT_DIR, f'wavegan_{dataset_name}.pt')
            torch.save({
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'epoch': epoch,
                'g_loss': avg_g,
            }, save_path)

        # Periodic save every 100 epochs
        if epoch % 100 == 0:
            period_path = os.path.join(CHECKPOINT_DIR, f'wavegan_{dataset_name}_ep{epoch}.pt')
            torch.save({
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'epoch': epoch,
                'g_loss': avg_g,
            }, period_path)
            print(f"  Saved periodic checkpoint: {period_path}")

    print(f"WaveGAN training complete. Best G_loss: {best_g_loss:.4f}")
    return model


def train_diffusion_model(model, model_name, X_train, y_train, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, lr=LR, dataset_name='bci2a', start_epoch=0):
    """Train a diffusion-based baseline model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} (starting from epoch {start_epoch})")
    print(f"{'='*60}")

    num_classes = int(y_train.max()) + 1
    fs = DATASET_CONFIG.get(dataset_name, {}).get('fs', 250)

    X = torch.FloatTensor(X_train).to(DEVICE)
    y = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - start_epoch)

    best_loss = float('inf')

    # For DiffEEGBooth, compute target laterality
    if isinstance(model, DiffEEGBooth):
        n_ch = X_train.shape[1]
        if dataset_name == 'bci2b':
            # BCI2b: 3 channels [C3, Cz, C4]
            c3_idx, c4_idx = 0, 2
        elif dataset_name == 'physionet':
            # PhysioNet: 64 channels, approximate C3/C4 indices
            c3_idx, c4_idx = 26, 30
        elif dataset_name == 'highgamma':
            # HighGamma: 128 channels, approximate C3/C4 indices
            c3_idx, c4_idx = 52, 60
        else:
            # BCI2a: 22 channels, C3=7, C4=11
            c3_idx, c4_idx = 7, 11
        # Safety check
        c3_idx = min(c3_idx, n_ch - 1)
        c4_idx = min(c4_idx, n_ch - 1)
        lat = []
        for c in range(num_classes):
            m = y_train == c
            if m.sum() == 0:
                lat.append(0.0)
                continue
            d = X_train[m]
            T = X_train.shape[-1]
            f = np.fft.rfftfreq(T, 1.0 / fs)
            am = (f >= 8) & (f <= 13)
            c3 = np.abs(np.fft.rfft(d[:, c3_idx, :])[:, am]) ** 2
            c4 = np.abs(np.fft.rfft(d[:, c4_idx, :])[:, am]) ** 2
            lat.append(float((c4.mean() - c3.mean()) / (c4.mean() + c3.mean() + 1e-10)))
        model.target_laterality = torch.tensor(lat, dtype=torch.float32, device=DEVICE)
        print(f"  Target laterality: {lat}")

    for epoch in range(1 + start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        loss_components = {}

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            if isinstance(model, BrainDiff):
                # BrainDiff: RL-guided loss weighting
                loss, loss_dict = model.loss(xb, yb)
            elif isinstance(model, DiffEEGBooth):
                # DiffEEGBooth: noise + spectral + ERD/ERS
                loss, loss_dict = model.loss(xb, yb)
            elif isinstance(model, CondDDPM):
                # Cond-DDPM: only noise loss
                loss = model.loss(xb, yb)
                loss_dict = {'noise': loss.item(), 'total': loss.item()}
            elif isinstance(model, EEGDiff):
                # EEGDiff: only noise loss
                loss = model.loss(xb, yb)
                loss_dict = {'noise': loss.item(), 'total': loss.item()}
            else:
                loss = model.loss(xb, yb)
                loss_dict = {'total': loss.item()}

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Accumulate loss components
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v if isinstance(v, float) else v

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Average loss components
        for k in loss_components:
            loss_components[k] /= n_batches

        if epoch % 10 == 0 or epoch == 1:
            comp_str = ', '.join([f"{k}={v:.4f}" for k, v in loss_components.items()])
            print(f"  Epoch {epoch}/{epochs}: {comp_str}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_{dataset_name}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'loss_components': loss_components,
            }, save_path)

        # Periodic save every 100 epochs (safety checkpoint)
        if epoch % 100 == 0:
            period_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_{dataset_name}_ep{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'loss_components': loss_components,
            }, period_path)
            print(f"  Saved periodic checkpoint: {period_path}")

    print(f"{model_name} training complete. Best loss: {best_loss:.4f}")
    return model


def train_cvae(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, dataset_name='bci2a', start_epoch=0):
    """Train CVAE model with KL warmup"""
    print(f"\n{'='*60}")
    print(f"Training CVAE (starting from epoch {start_epoch})")
    print(f"{'='*60}")

    X = torch.FloatTensor(X_train).to(DEVICE)
    y = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - start_epoch)

    best_loss = float('inf')
    warmup_epochs = (epochs - start_epoch) // 5  # Warmup KL over first 20% of remaining training
    target_beta = model.beta

    for epoch in range(1 + start_epoch, epochs + 1):
        # KL warmup: gradually increase beta from 0 to target
        if warmup_epochs > 0:
            epoch_in_warmup = epoch - start_epoch
            model.beta = target_beta * min(1.0, epoch_in_warmup / warmup_epochs)

        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            x_recon, z_mean, z_logvar = model(xb, yb)
            recon_loss = F.mse_loss(x_recon, xb, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            loss = recon_loss + model.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}, beta={model.beta:.5f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CHECKPOINT_DIR, f'cvae_{dataset_name}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, save_path)

        # Periodic save every 100 epochs
        if epoch % 100 == 0:
            period_path = os.path.join(CHECKPOINT_DIR, f'cvae_{dataset_name}_ep{epoch}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, period_path)
            print(f"  Saved periodic checkpoint: {period_path}")

    print(f"CVAE training complete. Best loss: {best_loss:.4f}")
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train baseline models for comparison')
    parser.add_argument('--dataset', type=str, default='bci2a',
                        choices=['bci2a', 'physionet', 'highgamma', 'bci2b'],
                        help='Dataset to train on')
    parser.add_argument('--model', type=str, default='wavegan',
                        choices=['wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth', 'cvae', 'all'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Get dataset config
    cfg = DATASET_CONFIG[args.dataset]
    channels = cfg['channels']
    n_samples = cfg['n_samples']
    fs = cfg['fs']
    num_classes = cfg['num_classes']

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset} (channels={channels}, n_samples={n_samples}, fs={fs}, classes={num_classes})")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    # Load data
    print(f"\nLoading {args.dataset} data...")
    X, y, subjects, sessions = cfg['load_fn']()

    # Use all data for training (evaluation scripts handle train/test splits)
    X_train = X
    y_train = y
    print(f"Training data: {X_train.shape}, labels: {np.bincount(y_train)}")

    models_to_train = ['wavegan', 'cond_ddpm', 'braindiff', 'eegdiff', 'diffeegbooth', 'cvae'] \
        if args.model == 'all' else [args.model]

    results = {}

    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# Training: {model_name} on {args.dataset}")
        print(f"{'#'*60}")

        start_time = time.time()

        if model_name == 'cvae':
            model = CVAE(channels=channels, latent_dim=64, out_length=n_samples, num_classes=num_classes)
        else:
            model = create_baseline_model(model_name, channels, n_samples, num_classes, fs)

        # Count parameters
        if model_name == 'wavegan':
            g_params = sum(p.numel() for p in model.generator.parameters())
            d_params = sum(p.numel() for p in model.discriminator.parameters())
            total_params = g_params + d_params
            print(f"Generator params: {g_params:,}, Discriminator params: {d_params:,}")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total params: {total_params:,}")

        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume and args.checkpoint:
            if os.path.exists(args.checkpoint):
                print(f"Loading checkpoint: {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
                if model_name == 'wavegan':
                    model.generator.load_state_dict(checkpoint['generator_state_dict'])
                    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Resuming from epoch {start_epoch + 1}")

        # Train (use dataset-specific checkpoint names)
        if model_name == 'wavegan':
            model = train_wavegan(model, X_train, y_train, args.epochs, args.batch_size, args.lr,
                                  dataset_name=args.dataset, start_epoch=start_epoch)
        elif model_name == 'cvae':
            model = train_cvae(model, X_train, y_train, args.epochs, args.batch_size, args.lr,
                               dataset_name=args.dataset, start_epoch=start_epoch)
        else:
            model = train_diffusion_model(model, model_name, X_train, y_train,
                                          args.epochs, args.batch_size, args.lr,
                                          dataset_name=args.dataset, start_epoch=start_epoch)

        elapsed = time.time() - start_time
        results[model_name] = {
            'params': total_params,
            'training_time': elapsed,
        }
        print(f"Training time: {elapsed/60:.1f} min")

    # Save results summary
    summary_path = os.path.join(CHECKPOINT_DIR, f'training_summary_{args.dataset}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
