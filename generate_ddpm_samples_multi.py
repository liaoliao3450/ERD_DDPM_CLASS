#!/usr/bin/env python3
"""为指定被试生成DDPM样本"""
import sys
import os
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'models', 'ddpm'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'utils'))
sys.path.insert(0, PROJECT_ROOT)

from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
)
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
C, T, NUM_CLASSES = 22, 1000, 4
FS = 250


def load_trained_ddpm(device=DEVICE):
    checkpoint_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'trained_ddpm.pt')
    if not os.path.exists(checkpoint_path):
        print(f"模型不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    eps_model = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(device)

    classifier_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'classifier_class_disc.pt')
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    if os.path.exists(classifier_path):
        clf_ckpt = torch.load(classifier_path, map_location=device)
        if isinstance(clf_ckpt, dict) and 'model_state_dict' in clf_ckpt:
            classifier.load_state_dict(clf_ckpt['model_state_dict'])
        else:
            classifier.load_state_dict(clf_ckpt)

    if isinstance(checkpoint, dict) and 'target_psd' in checkpoint:
        target_psd = checkpoint['target_psd'].to(device)
        target_laterality = checkpoint['target_laterality'].to(device)
    else:
        target_psd = torch.zeros(501).to(device)
        target_laterality = torch.zeros(4).to(device)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model, classifier=classifier,
        target_psd=target_psd, target_laterality=target_laterality,
        n_timesteps=1000, channels=C, n_samples=T, fs=FS
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        ddpm.load_state_dict(checkpoint['model'], strict=False)
    elif isinstance(checkpoint, dict) and 'eps_model' in checkpoint:
        ddpm.eps_model.load_state_dict(checkpoint['eps_model'], strict=False)
    else:
        ddpm.load_state_dict(checkpoint, strict=False)

    ddpm.eval()
    return ddpm


def generate_for_subject(subject_id):
    """为指定被试生成DDPM样本"""
    print(f"\n{'='*60}")
    print(f"为 Subject {subject_id} 生成DDPM样本")

    ddpm = load_trained_ddpm(DEVICE)
    if ddpm is None:
        print("无法加载DDPM模型，尝试直接无条件生成...")
        # 尝试无条件生成
        return

    # 加载真实数据
    X, y, subjects, sessions = load_bci2a_data()
    subj_idx = subject_id - 1  # 0-indexed
    mask = (subjects == subj_idx) & (sessions == 0)
    X_real = X[mask]
    y_real = y[mask]
    print(f"  真实数据: shape={X_real.shape}, 每类={[int((y_real==c).sum()) for c in range(4)]}")

    # 生成样本
    ddpm.eval()
    gen_X = []
    gen_y = []

    with torch.no_grad():
        for c in range(4):
            class_mask = y_real == c
            n_samples = int(class_mask.sum())
            if n_samples == 0:
                continue
            print(f"  生成类别 {c}: {n_samples} 个样本...")
            yg = torch.full((n_samples,), c, dtype=torch.long, device=DEVICE)
            samples = ddpm.sample_ddim(n_samples, yg, steps=50, guidance_scale=5.0, device=DEVICE)
            gen_X.append(samples.cpu().numpy())
            gen_y.extend([c] * n_samples)

    gen_X = np.concatenate(gen_X, axis=0)
    gen_y = np.array(gen_y)
    print(f"  生成数据: shape={gen_X.shape}, 每类={[int((gen_y==c).sum()) for c in range(4)]}")

    # 全局标准化对齐
    real_mean = X_real.mean(axis=(0, 2), keepdims=True)
    real_std = X_real.std(axis=(0, 2), keepdims=True)
    gen_mean = gen_X.mean(axis=(0, 2), keepdims=True)
    gen_std = gen_X.std(axis=(0, 2), keepdims=True)
    gen_X_matched = (gen_X - gen_mean) / (gen_std + 1e-8) * real_std + real_mean

    # 保存
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'ddpm_samples')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'ddpm_samples_subject{subject_id}.npy'), gen_X_matched)
    np.save(os.path.join(output_dir, f'ddpm_labels_subject{subject_id}.npy'), gen_y)
    print(f"  已保存: ddpm_samples_subject{subject_id}.npy, ddpm_labels_subject{subject_id}.npy")


if __name__ == '__main__':
    for sid in [6, 3]:
        generate_for_subject(sid)
