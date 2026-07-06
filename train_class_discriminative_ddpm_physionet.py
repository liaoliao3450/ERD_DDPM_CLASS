#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-Discriminative DDPM 训练脚本 - PhysioNet MI4C专用

数据加载: 使用 DTTD 的 PhysioNetMIDataset 类 (与参照工程完全一致)
  - 从 EDF 原始文件加载
  - 64 通道 (基于 10-10 系统)
  - 预处理: bandpass 4-30Hz, notch 50Hz, CAR, resample 160Hz
  - 4类: left hand (0), right hand (1), both fists (2), both feet (3)
  - 640 时间步 (4秒 @ 160Hz)

依赖:
- DTTD-DDPM/data/physionet_mi.py (数据加载)
- E:/data/PhysioNetMI (原始 EDF 数据)
"""
import os
import sys
import io
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# 添加 DTTD 项目路径 (使用 DTTD 的数据加载类，确保数据预处理完全一致)
DTTD_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "DTTD-DDPM")
if os.path.isdir(DTTD_ROOT):
    sys.path.insert(0, DTTD_ROOT)
    from data.physionet_mi import PhysioNetMIDataset, INPUT_CHANNEL_INDICES_16
    USE_DTTD_LOADER = True
    print(f"[INFO] DTTD 项目可用: {DTTD_ROOT}")
else:
    USE_DTTD_LOADER = False
    print(f"[WARN] DTTD 项目未找到")

# 始终使用 PhysioNetMI4C 数据（与评估脚本一致）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from data_loader_physionet_mi4c import load_physionet_mi4c_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core/models/ddpm"))
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    pretrain_classifier
)

# ============================================================================
# 配置
# ============================================================================
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据参数 - PhysioNet MI4C (与 DTTD 一致)
CHANNELS = 64
N_SAMPLES = 640  # 4秒 @ 160Hz
FS = 160
NUM_CLASSES = 4

# 训练参数
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# 损失权重 (与BCI2a/2b一致)
NOISE_WEIGHT = 1.0
SPECTRAL_WEIGHT = 0.5
ERD_WEIGHT = 0.5
CLS_WEIGHT = 1.0

# 分类器引导
GUIDANCE_SCALE = 7.0
CLASSIFIER_PRETRAIN_EPOCHS = 200


def compute_targets(X: np.ndarray, y: np.ndarray, fs: int, c3_idx: int, c4_idx: int):
    """计算 target_psd (全局) 与 target_laterality (按类别)"""
    T = int(X.shape[-1])
    num_classes = int(len(np.unique(y)))

    # target PSD: 全局计算 (与BCI2a/2b一致) [num_freqs]
    fft = np.fft.rfft(X, axis=-1)
    psd = (np.abs(fft) ** 2).mean(axis=(0, 1)).astype(np.float32)

    # target laterality: per class alpha-band laterality (8-13Hz)
    lat = []
    for c in range(num_classes):
        m = y == c
        if int(m.sum()) == 0:
            lat.append(0.0)
            continue
        d = X[m]
        f = np.fft.rfftfreq(T, 1.0 / fs)
        am = (f >= 8) & (f <= 13)
        c3 = np.abs(np.fft.rfft(d[:, c3_idx, :])[:, am]) ** 2
        c4 = np.abs(np.fft.rfft(d[:, c4_idx, :])[:, am]) ** 2
        lat.append(float((c4.mean() - c3.mean()) / (c4.mean() + c3.mean() + 1e-10)))

    return torch.tensor(psd, dtype=torch.float32), torch.tensor(lat, dtype=torch.float32)


def load_all_subjects_with_cache(data_path, cache_path='paper_results/physionet_mi/physionet_mi_all_subjects.npz'):
    """加载所有被试数据 (与 DTTD physionet_mi_classification.py 一致)"""
    # 优先从预提取的npz文件加载
    npy_candidates = [
        os.path.join(data_path, 'physionet_mi_preprocessed.npz'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data_cache', 'physionet_mi_preprocessed.npz'),
    ]
    npy_path = None
    for p in npy_candidates:
        if os.path.exists(p):
            npy_path = p
            break
    if npy_path is not None:
        print(f"[预提取] 从npy加载: {npy_path}")
        t0 = __import__('time').time()
        npz = np.load(npy_path, allow_pickle=True)
        data_arr = npz['data']
        labels_arr = npz['labels']
        sid_arr = npz['subject_ids']
        subject_data_map = {}
        valid_subject_ids = []
        for i, sid_str in enumerate(sid_arr):
            sid_int = int(sid_str[1:])  # 'S001' -> 1
            subject_data_map[sid_int] = (data_arr[i], labels_arr[i])
            valid_subject_ids.append(sid_int)
        print(f"[预提取] 加载完成: {len(valid_subject_ids)} 个被试, 耗时 {__import__('time').time()-t0:.1f}s")
        return valid_subject_ids, subject_data_map

    # 其次从缓存加载
    if os.path.exists(cache_path):
        print(f"[缓存] 从缓存加载: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        subject_data_map = dict(cached['subject_data_map'].item())
        valid_subject_ids = list(subject_data_map.keys())
        print(f"[缓存] 加载完成: {len(valid_subject_ids)} 个被试")
        return valid_subject_ids, subject_data_map

    print("[缓存] 未找到缓存，从EDF文件加载...")
    all_subject_ids = list(range(1, 110))
    valid_subject_ids = []
    subject_data_map = {}

    for sid in all_subject_ids:
        try:
            dataset = PhysioNetMIDataset(
                data_path=data_path,
                subject_ids=[sid],
                runs={4, 6, 8, 10, 12, 14},
                reconstruction_mode=True
            )
            if len(dataset.data) > 0:
                subject_data_map[sid] = (dataset.data, dataset.labels)
                valid_subject_ids.append(sid)
                print(f"  被试{sid}: {len(dataset.data)} trials")
        except Exception as e:
            pass

    # 保存缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, subject_data_map=subject_data_map)
    print(f"[缓存] 已保存至: {cache_path} ({len(valid_subject_ids)} 个被试)")

    return valid_subject_ids, subject_data_map


def main():
    parser = argparse.ArgumentParser(description="PhysioNet MI4C ERD-DDPM 训练")
    parser.add_argument("--data_path", type=str, default="E:/data/PhysioNetMI",
                        help="PhysioNet MI 原始数据路径 (包含 EDF 文件)")
    args = parser.parse_args()

    print("=" * 70)
    print("Class-Discriminative DDPM 训练 - PhysioNet MI4C")
    print("数据加载: 使用 DTTD PhysioNetMIDataset (与参照工程一致)")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"数据路径: {args.data_path}")
    print(f"检查点目录: {CHECKPOINT_DIR}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据 (与 DTTD 一致, 带缓存)
    print("\n步骤1: 加载 PhysioNet MI4C 数据 (使用 PhysioNetMI4C 预处理数据，与评估一致)...")
    data_root = os.path.join(os.path.dirname(__file__), "..", "..", "data/processed/PhysioNetMI4C")
    X, y, subjects, sessions, _ = load_physionet_mi4c_data(data_root=data_root)
    print(f"  数据形状: {X.shape}")
    print(f"  标签分布: {np.bincount(y)}")
    print(f"  被试数量: {len(np.unique(subjects))}")

    # 数据预处理 (与 DTTD 一致: 逐通道标准化)
    print("\n步骤2: 数据预处理...")
    # 逐通道计算均值和标准差 (与DTTD physionet_mi_classification.py一致)
    data_mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # [1, C, 1]
    data_std = X.std(axis=(0, 2), keepdims=True).astype(np.float32)    # [1, C, 1]
    data_std = np.maximum(data_std, 1e-6)
    X_norm = ((X - data_mean) / data_std).astype(np.float32)
    # Clamp极端值 (与DTTD生成数据时clip一致, 防止outlier影响训练)
    clip_count = np.sum(np.abs(X_norm) > 5)
    X_norm = np.clip(X_norm, -5.0, 5.0)
    print(f"  逐通道标准化: mean={data_mean.mean():.6f}, std={data_std.mean():.6f}")
    print(f"  标准化后数据范围: [{X_norm.min():.4f}, {X_norm.max():.4f}] (clipped {clip_count} values > 5)")

    # 计算target_psd和target_laterality (与 train_ddpm_once 一致)
    print("\n步骤2.5: 计算 target PSD 和 ERD laterality...")
    # PhysioNet 64通道中 C3=28, C4=32 (从CHANNEL_NAMES_64推断)
    c3_idx, c4_idx = 28, 32
    target_psd, target_laterality = compute_targets(X_norm, y, FS, c3_idx, c4_idx)
    print(f"  target_psd: shape={target_psd.shape}, range=[{target_psd.min():.4f}, {target_psd.max():.4f}]")
    print(f"  target_laterality: {target_laterality.tolist()}")

    # 创建数据集
    dataset = TensorDataset(torch.FloatTensor(X_norm), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 创建模型
    print("\n步骤3: 创建模型...")
    eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES, base_ch=64).to(DEVICE)
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(DEVICE)

    # 预训练分类器
    print("\n步骤4: 预训练分类器...")
    classifier_path = os.path.join(CHECKPOINT_DIR, 'classifier_physionet.pt')

    if os.path.exists(classifier_path):
        print(f"  加载预训练分类器: {classifier_path}")
        checkpoint = torch.load(classifier_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            classifier.load_state_dict(checkpoint)
        classifier.eval()
    else:
        print(f"  训练分类器 ({CLASSIFIER_PRETRAIN_EPOCHS} epochs)...")
        classifier = pretrain_classifier(
            classifier,
            torch.FloatTensor(X_norm).to(DEVICE),
            torch.LongTensor(y).to(DEVICE),
            epochs=CLASSIFIER_PRETRAIN_EPOCHS,
            batch_size=64,
            lr=1e-3,
            device=DEVICE,
            save_path=classifier_path,
            verbose=True
        )

    # 创建 DDPM
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd.to(DEVICE),
        target_laterality=target_laterality.to(DEVICE),
        n_timesteps=1000,
        channels=CHANNELS,
        n_samples=N_SAMPLES,
        fs=FS,
    ).to(DEVICE)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(ddpm.eps_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 断点续训
    start_epoch = 1
    best_loss = float('inf')
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_class_discriminative_physionet_mi4c.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # 检查模型维度是否匹配
        saved_in_proj = ckpt.get('eps_model.in_proj.weight') if isinstance(ckpt, dict) and 'eps_model.in_proj.weight' in ckpt else ckpt.get('model_state_dict', {}).get('eps_model.in_proj.weight')
        current_in_proj_shape = ddpm.eps_model.in_proj.weight.shape
        # 检查target_psd维度
        saved_psd = ckpt.get('model_state_dict', {}).get('target_psd')
        current_psd_shape = ddpm.target_psd.shape
        psd_match = saved_psd is None or saved_psd.shape == current_psd_shape
        if saved_in_proj is not None and saved_in_proj.shape == current_in_proj_shape and psd_match:
            if 'model_state_dict' in ckpt:
                ddpm.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                best_loss = ckpt.get('best_loss', float('inf'))
            else:
                print(f"  旧格式checkpoint, 重新训练")
            for _ in range(start_epoch - 1):
                scheduler.step()
            if start_epoch > 1:
                print(f"  断点续训: 从epoch {start_epoch}继续, best_loss={best_loss:.4f}")
        else:
            print(f"  已有checkpoint不兼容, 重新训练")

    # 训练
    print("\n步骤5: 开始训练 DDPM...")
    print(f"  训练轮数: {EPOCHS} (从epoch {start_epoch}开始)")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  损失权重: noise={NOISE_WEIGHT}, spectral={SPECTRAL_WEIGHT}, erd={ERD_WEIGHT}, cls={CLS_WEIGHT}")

    patience = 0
    max_patience = 30
    last_loss_dict = {}

    for ep in range(start_epoch, EPOCHS + 1):
        ddpm.train()
        total_loss = 0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            loss, loss_dict = ddpm.loss(
                xb, yb,
                noise_weight=NOISE_WEIGHT,
                spectral_weight=SPECTRAL_WEIGHT,
                erd_weight=ERD_WEIGHT,
                cls_weight=CLS_WEIGHT,
                low_t_threshold=400  # 40%时间步计算辅助损失
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            last_loss_dict = loss_dict

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:4d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"noise={last_loss_dict.get('noise',0):.4f} spec={last_loss_dict.get('spectral',0):.4f} "
                  f"erd={last_loss_dict.get('erd',0):.4f} cls={last_loss_dict.get('classification',0):.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            save_path = os.path.join(CHECKPOINT_DIR, 'best_class_discriminative_physionet_mi4c.pt')
            torch.save({
                'model_state_dict': ddpm.state_dict(),
                'target_psd': ddpm.target_psd.cpu(),
                'target_laterality': ddpm.target_laterality.cpu(),
                'epoch': ep,
                'best_loss': best_loss,
                'channels': CHANNELS,
                'n_samples': N_SAMPLES,
                'fs': FS,
                'num_classes': NUM_CLASSES,
                'data_mean': data_mean,
                'data_std': data_std,
                'data_loader': 'PhysioNetMI4C',
            }, save_path)
            if ep % 10 == 0:
                print(f"  [OK] 保存最佳模型到 {save_path}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n  早停: {max_patience} 轮没有改善")
                break

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"模型保存到: {CHECKPOINT_DIR}/best_class_discriminative_physionet_mi4c.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
