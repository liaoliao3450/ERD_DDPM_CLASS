"""
统一生成所有方法的样本并保存到磁盘

生成方法:
  1. CVAE
  2. WaveGAN
  3. Cond-DDPM
  4. BrainDiff
  5. EEGDiff
  6. DiffEEGBooth
  7. DDPM (Ours)

保存格式:
  outputs/samples/{dataset}/{method}_samples.npy
  outputs/samples/{dataset}/{method}_labels.npy

后续画图和算指标时直接加载这些 .npy 文件，无需重复生成。
模型调试好后重新运行此脚本即可更新样本。
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import signal as scipy_signal

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'ddpm'))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'baselines'))

from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
)
from comparison_models import WaveGAN, CondDDPM, BrainDiff, EEGDiff, DiffEEGBooth
from cvae_gaussian import CVAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集配置
DATASET_CONFIG = {
    'bci2a': {
        'data_dir': 'data/processed/BCI2a',
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
        'fs': 250,
        'c3_idx': 7,
        'c4_idx': 11,
        'n_per_session': 288,
        'n_subjects': 9,
    },
    'bci2b': {
        'data_dir': 'data/processed/BCI2b',
        'channels': 3,
        'n_samples': 1000,
        'num_classes': 4,
        'fs': 250,
        'c3_idx': 0,
        'c4_idx': 2,
        'n_per_session': 72,
        'n_subjects': 9,
    },
    'physionet': {
        'data_dir': 'data/processed/PhysioNet',
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
        'fs': 200,
        'c3_idx': 7,
        'c4_idx': 11,
        'n_per_session': 225,
        'n_subjects': 9,
    },
}


# ============================================================================
# 数据加载 (与训练脚本一致)
# ============================================================================

def load_data(dataset='bci2a'):
    """加载数据并做全局标准化 (与 UMAP 脚本一致)"""
    cfg = DATASET_CONFIG[dataset]
    C, T, NC = cfg['channels'], cfg['n_samples'], cfg['num_classes']
    n_per_session = cfg['n_per_session']
    n_subjects = cfg['n_subjects']

    print(f"\n加载 {dataset} 数据...")
    X = np.load(f"{cfg['data_dir']}/X.npy") * 1e6  # 转微伏
    y = np.load(f"{cfg['data_dir']}/y.npy")

    if dataset == 'bci2a':
        sess_ids = np.tile(np.repeat([0, 1], n_per_session), n_subjects)
        mask = sess_ids == 0
        X_all = X[mask]
        X_train = X_all[:n_per_session]
        y_train = (y[mask][:n_per_session] - 1).astype(np.int64)
    elif dataset == 'bci2b':
        # BCI2b 每个受试者 session 不同，取所有 session 0
        sess_ids = np.tile(np.repeat([0, 1]), n_subjects)
        mask = sess_ids == 0
        X_all = X[mask]
        X_train = X_all[:n_per_session]
        y_train = (y[mask][:n_per_session] - 1).astype(np.int64)
    else:  # physionet
        X_train = X[:n_per_session]
        y_train = (y[:n_per_session]).astype(np.int64)

    # 全局标准化 (与 visualize_plot_umap_all_methods.py 一致)
    X_mean = X_train.mean() if dataset != 'bci2a' else X_all.mean()
    X_std = X_train.std() if dataset != 'bci2a' else X_all.std()
    X_norm = ((X_train - X_mean) / X_std).astype(np.float32)

    print(f"  数据形状: {X_norm.shape}")
    print(f"  类别分布: {np.bincount(y_train)}")
    print(f"  标准化: mean={X_mean:.4f}, std={X_std:.4f}")
    print(f"  标准化后范围: [{X_norm.min():.4f}, {X_norm.max():.4f}]")

    return X_norm, y_train, X_mean, X_std


def compute_target_psd(X, device=DEVICE):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd.to(device)


def compute_class_laterality(X, y, cfg):
    """计算每个类别的平均侧化指数"""
    NC = cfg['num_classes']
    FS = cfg['fs']
    C3_IDX = cfg['c3_idx']
    C4_IDX = cfg['c4_idx']

    laterality = torch.zeros(NC)
    for cls in range(NC):
        cls_data = X[y == cls]
        lat_values = []
        for i in range(len(cls_data)):
            f, psd_c3 = scipy_signal.welch(cls_data[i, C3_IDX], fs=FS, nperseg=256)
            f, psd_c4 = scipy_signal.welch(cls_data[i, C4_IDX], fs=FS, nperseg=256)
            alpha_mask = (f >= 8) & (f <= 13)
            c3_alpha = psd_c3[alpha_mask].mean()
            c4_alpha = psd_c4[alpha_mask].mean()
            lat = (c4_alpha - c3_alpha) / (c4_alpha + c3_alpha + 1e-10)
            lat_values.append(lat)
        laterality[cls] = float(np.mean(lat_values)) if lat_values else 0.0
    return laterality


# ============================================================================
# 模型加载
# ============================================================================

def load_baseline(model_class, ckpt_path, cfg, device=DEVICE):
    """加载基线扩散模型"""
    model = model_class(
        channels=cfg['channels'],
        n_samples=cfg['n_samples'],
        num_classes=cfg['num_classes'],
        fs=cfg['fs'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_wavegan(ckpt_path, cfg, device=DEVICE):
    """加载 WaveGAN"""
    model = WaveGAN(
        channels=cfg['channels'],
        out_length=cfg['n_samples'],
        num_classes=cfg['num_classes'],
        z_dim=100,
        fs=cfg['fs'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_cvae(ckpt_path, cfg, device=DEVICE):
    """加载 CVAE"""
    model = CVAE(
        channels=cfg['channels'],
        latent_dim=64,
        out_length=cfg['n_samples'],
        num_classes=cfg['num_classes'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_ddpm(X_train, y_train, cfg, device=DEVICE,
              checkpoint_path='checkpoints/best_class_discriminative.pt'):
    """加载 DDPM (Ours)"""
    print(f"  加载 DDPM: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    target_psd = compute_target_psd(X_train, device)
    target_laterality = compute_class_laterality(X_train, y_train, cfg).to(device)

    eps_model = MultiScaleCondUNet(
        channels=cfg['channels'],
        num_classes=cfg['num_classes'],
    ).to(device)

    # 加载分类器
    classifier = EEGClassifier(
        channels=cfg['channels'],
        n_samples=cfg['n_samples'],
        num_classes=cfg['num_classes'],
    ).to(device)
    clf_path = 'checkpoints/classifier_class_disc.pt'
    if os.path.exists(clf_path):
        clf_ckpt = torch.load(clf_path, map_location=device, weights_only=False)
        if isinstance(clf_ckpt, dict) and 'model_state_dict' in clf_ckpt:
            classifier.load_state_dict(clf_ckpt['model_state_dict'])
        else:
            classifier.load_state_dict(clf_ckpt)

    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=cfg['channels'],
        n_samples=cfg['n_samples'],
        fs=cfg['fs'],
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ddpm.load_state_dict(checkpoint['model_state_dict'])
    else:
        ddpm.load_state_dict(checkpoint)

    ddpm.eval()
    return ddpm


# ============================================================================
# 样本生成
# ============================================================================

def generate_diffusion_samples(model, n_per_class, use_ddim=True, ddim_steps=50):
    """生成扩散模型样本 (Cond-DDPM, BrainDiff, EEGDiff, DiffEEGBooth)"""
    gen_data, gen_labels = [], []
    for cls in range(model.num_classes if hasattr(model, 'num_classes') else 4):
        y = torch.full((n_per_class,), cls, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if use_ddim and hasattr(model, 'sample_ddim'):
                samples = model.sample_ddim(n_per_class, y, steps=ddim_steps, device=DEVICE)
            else:
                samples = model.sample(n_per_class, y, device=DEVICE)
        gen_data.append(samples.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
    return np.concatenate(gen_data), np.array(gen_labels)


def generate_wavegan_samples(model, n_per_class, num_classes=4):
    """生成 WaveGAN 样本"""
    gen_data, gen_labels = [], []
    for cls in range(num_classes):
        y = torch.full((n_per_class,), cls, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            samples = model.generate(n_per_class, y, device=DEVICE)
        gen_data.append(samples.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
    return np.concatenate(gen_data), np.array(gen_labels)


def generate_cvae_samples(model, n_per_class, num_classes=4):
    """生成 CVAE 样本"""
    gen_data, gen_labels = [], []
    for cls in range(num_classes):
        y = torch.full((n_per_class,), cls, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            samples = model.generate(n_per_class, y, device=DEVICE)
        gen_data.append(samples.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
    return np.concatenate(gen_data), np.array(gen_labels)


def align_to_real(gen_X, X_real, clip=False):
    """将生成数据的统计量对齐到真实数据 (按通道 mean/std 匹配)

    按通道对齐保留了通道间差异 (如 C3/C4 的侧化模式)，
    比全局对齐更合理。

    Args:
        gen_X: 生成数据 [N, C, T]
        X_real: 真实数据 [N, C, T]
        clip: 是否 clip 到 [-5, 5] (默认不 clip，保留完整动态范围)
    """
    # 按通道计算统计量 (保留通道间差异)
    real_mean = X_real.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    real_std = X_real.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
    gen_mean = gen_X.mean(axis=(0, 2), keepdims=True)    # (1, C, 1)
    gen_std = gen_X.std(axis=(0, 2), keepdims=True)      # (1, C, 1)

    # 标准化后缩放到 real 的尺度 (按通道)
    gen_X_aligned = ((gen_X - gen_mean) / (gen_std + 1e-8)) * real_std + real_mean

    if clip:
        gen_X_aligned = np.clip(gen_X_aligned, -5.0, 5.0)
    return gen_X_aligned.astype(np.float32)


def generate_ddpm_samples(ddpm, n_per_class, guidance_scale=0.1, num_classes=4):
    """生成 DDPM (Ours) 样本"""
    gen_data, gen_labels = [], []
    import time
    for cls in range(num_classes):
        t0 = time.time()
        print(f"    生成类别 {cls}/{num_classes-1} (n={n_per_class})...", end='', flush=True)
        y = torch.full((n_per_class,), cls, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            samples = ddpm.sample(n_per_class, y, guidance_scale=guidance_scale, device=DEVICE)
        gen_data.append(samples.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
        print(f" 完成 ({time.time()-t0:.1f}s)", flush=True)
    gen_X = np.concatenate(gen_data)
    print(f"    DDPM gen stats: min={gen_X.min():.4f}, max={gen_X.max():.4f}, "
          f"mean={gen_X.mean():.4f}, std={gen_X.std():.4f}")
    return gen_X, np.array(gen_labels)


# ============================================================================
# 保存
# ============================================================================

def save_samples(X, y, dataset, method, output_dir='outputs/samples'):
    """保存样本到磁盘"""
    save_dir = os.path.join(output_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'{method}_samples.npy'), X)
    np.save(os.path.join(save_dir, f'{method}_labels.npy'), y)
    print(f"  ✅ 保存: {save_dir}/{method}_samples.npy {X.shape}")


def load_samples(dataset, method, output_dir='outputs/samples'):
    """从磁盘加载样本"""
    save_dir = os.path.join(output_dir, dataset)
    X = np.load(os.path.join(save_dir, f'{method}_samples.npy'))
    y = np.load(os.path.join(save_dir, f'{method}_labels.npy'))
    return X, y


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生成所有方法的样本并保存')
    parser.add_argument('--dataset', type=str, default='bci2a',
                        choices=['bci2a', 'bci2b', 'physionet'],
                        help='数据集')
    parser.add_argument('--n_per_class', type=int, default=100,
                        help='每类生成样本数')
    parser.add_argument('--guidance_scale', type=float, default=0.1,
                        help='DDPM 分类器引导强度')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM 采样步数')
    parser.add_argument('--methods', type=str, nargs='+', default='all',
                        help='要生成的方法 (默认 all)')
    parser.add_argument('--output_dir', type=str, default='outputs/samples',
                        help='输出目录')
    args = parser.parse_args()

    print("=" * 70)
    print("生成所有方法的样本")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"数据集: {args.dataset}")
    print(f"每类样本数: {args.n_per_class}")
    print(f"DDPM guidance_scale: {args.guidance_scale}")

    cfg = DATASET_CONFIG[args.dataset]
    all_methods = ['cvae', 'wavegan', 'cond_ddpm', 'braindiff',
                   'eegdiff', 'diffeegbooth', 'ddpm']
    methods = all_methods if args.methods == 'all' else args.methods

    # 加载数据
    X_train, y_train, X_mean, X_std = load_data(args.dataset)

    # 不做 clip，保留完整动态范围 (与 visualize_class_discriminative_ddpm_tsne.py 一致)
    print(f"  Real 数据: std={X_train.std():.4f}, 范围: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # 基线模型 checkpoint 路径
    baseline_ckpt = {
        'cvae':         f'checkpoints/baselines/cvae_{args.dataset}.pt',
        'wavegan':      f'checkpoints/baselines/wavegan_{args.dataset}.pt',
        'cond_ddpm':    f'checkpoints/baselines/cond_ddpm_{args.dataset}.pt',
        'braindiff':    f'checkpoints/baselines/braindiff_{args.dataset}.pt',
        'eegdiff':      f'checkpoints/baselines/eegdiff_{args.dataset}.pt',
        'diffeegbooth': f'checkpoints/baselines/diffeegbooth_{args.dataset}.pt',
    }

    # ---- 生成 CVAE ----
    if 'cvae' in methods:
        print(f"\n[{1}/{len(methods)}] 生成 CVAE 样本...")
        try:
            model = load_cvae(baseline_ckpt['cvae'], cfg)
            X, y = generate_cvae_samples(model, args.n_per_class, cfg['num_classes'])
            # 不做 align_to_real，保留原始类别信息 (与 tsne 脚本一致)
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'cvae', args.output_dir)
        except Exception as e:
            print(f"  ❌ CVAE 生成失败: {e}")

    # ---- 生成 WaveGAN ----
    if 'wavegan' in methods:
        print(f"\n[{2}/{len(methods)}] 生成 WaveGAN 样本...")
        try:
            model = load_wavegan(baseline_ckpt['wavegan'], cfg)
            X, y = generate_wavegan_samples(model, args.n_per_class, cfg['num_classes'])
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'wavegan', args.output_dir)
        except Exception as e:
            print(f"  ❌ WaveGAN 生成失败: {e}")

    # ---- 生成 Cond-DDPM ----
    if 'cond_ddpm' in methods:
        print(f"\n[{3}/{len(methods)}] 生成 Cond-DDPM 样本...")
        try:
            model = load_baseline(CondDDPM, baseline_ckpt['cond_ddpm'], cfg)
            X, y = generate_diffusion_samples(model, args.n_per_class,
                                              use_ddim=True, ddim_steps=args.ddim_steps)
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'cond_ddpm', args.output_dir)
        except Exception as e:
            print(f"  ❌ Cond-DDPM 生成失败: {e}")

    # ---- 生成 BrainDiff ----
    if 'braindiff' in methods:
        print(f"\n[{4}/{len(methods)}] 生成 BrainDiff 样本...")
        try:
            model = load_baseline(BrainDiff, baseline_ckpt['braindiff'], cfg)
            X, y = generate_diffusion_samples(model, args.n_per_class,
                                              use_ddim=True, ddim_steps=args.ddim_steps)
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'braindiff', args.output_dir)
        except Exception as e:
            print(f"  ❌ BrainDiff 生成失败: {e}")

    # ---- 生成 EEGDiff ----
    if 'eegdiff' in methods:
        print(f"\n[{5}/{len(methods)}] 生成 EEGDiff 样本...")
        try:
            model = load_baseline(EEGDiff, baseline_ckpt['eegdiff'], cfg)
            X, y = generate_diffusion_samples(model, args.n_per_class,
                                              use_ddim=True, ddim_steps=args.ddim_steps)
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'eegdiff', args.output_dir)
        except Exception as e:
            print(f"  ❌ EEGDiff 生成失败: {e}")

    # ---- 生成 DiffEEGBooth ----
    if 'diffeegbooth' in methods:
        print(f"\n[{6}/{len(methods)}] 生成 DiffEEGBooth 样本...")
        try:
            model = load_baseline(DiffEEGBooth, baseline_ckpt['diffeegbooth'], cfg)
            X, y = generate_diffusion_samples(model, args.n_per_class,
                                              use_ddim=True, ddim_steps=args.ddim_steps)
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'diffeegbooth', args.output_dir)
        except Exception as e:
            print(f"  ❌ DiffEEGBooth 生成失败: {e}")

    # ---- 生成 DDPM (Ours) ----
    if 'ddpm' in methods:
        print(f"\n[{7}/{len(methods)}] 生成 DDPM (Ours) 样本...")
        try:
            ddpm = load_ddpm(X_train, y_train, cfg)
            X, y = generate_ddpm_samples(ddpm, args.n_per_class,
                                         guidance_scale=args.guidance_scale,
                                         num_classes=cfg['num_classes'])
            # 不做 align_to_real，保留原始类别信息
            print(f"  形状: {X.shape}, 范围: [{X.min():.3f}, {X.max():.3f}], std={X.std():.3f}")
            save_samples(X, y, args.dataset, 'ddpm', args.output_dir)
        except Exception as e:
            print(f"  ❌ DDPM 生成失败: {e}")

    print("\n" + "=" * 70)
    print("样本生成完成！")
    print("=" * 70)
    print(f"\n保存目录: {args.output_dir}/{args.dataset}/")
    print("\n后续可直接加载使用:")
    print(f"  X = np.load('{args.output_dir}/{args.dataset}/ddpm_samples.npy')")
    print(f"  y = np.load('{args.output_dir}/{args.dataset}/ddpm_labels.npy')")
    print("\n可用方法: cvae, wavegan, cond_ddpm, braindiff, eegdiff, diffeegbooth, ddpm")


if __name__ == '__main__':
    main()
