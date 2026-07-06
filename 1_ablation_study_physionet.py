"""
消融实验 - PhysioNet MI4C 数据集
测试每个组件的贡献：ERD约束、分类器引导、频谱约束
使用LMSO 10-Fold Cross-Subject方式
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core", "models", "ddpm"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core", "models", "baselines"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "utils"))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import json
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from class_discriminative import (
    ClassDiscriminativeDDPM, MultiScaleCondUNet, EEGClassifier, pretrain_classifier
)
from data_loader_physionet_mi4c import load_physionet_mi4c_data

# PhysioNet MI4C 数据配置
CHANNELS = 64
N_SAMPLES = 640
NUM_CLASSES = 4
FS = 160

# PhysioNet 64通道标准10-10系统中C3/C4的索引
C3_IDX = 20
C4_IDX = 28

# 分类器训练超参数
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3

N_FOLDS = 10
AUG_RATIO = 0.2


def load_physionet_data(data_root="data/processed/PhysioNetMI4C"):
    """加载PhysioNet数据"""
    X, y, subjects, sessions, _ = load_physionet_mi4c_data(data_root=data_root)
    
    # 过滤到4类
    mask = y < NUM_CLASSES
    X, y = X[mask], y[mask]
    subjects, sessions = subjects[mask], sessions[mask]
    
    # 标准化
    data_mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    data_std = np.maximum(X.std(axis=(0, 2), keepdims=True).astype(np.float32), 1e-6)
    X = (X - data_mean) / data_std
    
    return X, y, subjects, sessions


def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd

def compute_class_laterality(X, y, num_classes=NUM_CLASSES, fs=FS, c3_idx=C3_IDX, c4_idx=C4_IDX):
    """计算每个类别的平均侧化指数"""
    laterality = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        cls_data = X[y == cls]
        if len(cls_data) == 0:
            laterality[cls] = 0.0
            continue
            
        lat_values = []
        for i in range(len(cls_data)):
            f, psd_c3 = signal.welch(cls_data[i, c3_idx], fs=fs, nperseg=min(256, N_SAMPLES))
            f, psd_c4 = signal.welch(cls_data[i, c4_idx], fs=fs, nperseg=min(256, N_SAMPLES))
            alpha_mask = (f >= 8) & (f <= 13)
            c3_alpha = psd_c3[alpha_mask].mean()
            c4_alpha = psd_c4[alpha_mask].mean()
            lat = (c4_alpha - c3_alpha) / (c4_alpha + c3_alpha + 1e-10)
            lat_values.append(lat)
        
        laterality[cls] = float(np.mean(lat_values)) if lat_values else 0.0
    
    return laterality

def normalize_generated_data_to_real_stats(X_real: np.ndarray, X_gen: np.ndarray) -> np.ndarray:
    """将生成数据的统计特性对齐到真实数据"""
    X_real = X_real.astype(np.float32)
    X_gen = X_gen.astype(np.float32)

    real_mean = X_real.mean(axis=(0, 2), keepdims=True)
    real_std = X_real.std(axis=(0, 2), keepdims=True)
    gen_mean = X_gen.mean(axis=(0, 2), keepdims=True)
    gen_std = X_gen.std(axis=(0, 2), keepdims=True)

    eps = 1e-8
    if float(np.max(gen_std)) < eps:
        print("  ⚠️  生成数据逐通道标准差过小，跳过对齐")
        return X_gen

    X_gen_norm = (X_gen - gen_mean) / (gen_std + eps)
    X_gen_aligned = X_gen_norm * (real_std + eps) + real_mean
    return X_gen_aligned

def _float_to_tag(v: float) -> str:
    return str(v).replace('.', 'p')

def get_ablation_ckpt_path(variant_config: dict, fold_idx: int = -1) -> str:
    cfg = variant_config['config']
    erd_tag = _float_to_tag(cfg.get('erd_weight', 0.0))
    cls_tag = _float_to_tag(cfg.get('cls_weight', 0.0))
    spec_tag = _float_to_tag(cfg.get('spectral_weight', 0.0))
    guid_tag = _float_to_tag(cfg.get('guidance_scale', 0.0))

    ckpt_dir = os.path.join('checkpoints', 'ablation_models_physionet')
    os.makedirs(ckpt_dir, exist_ok=True)
    if fold_idx >= 0:
        filename = f"ddpm_erd{erd_tag}_cls{cls_tag}_spec{spec_tag}_guid{guid_tag}_fold{fold_idx}.pt"
    else:
        filename = f"ddpm_erd{erd_tag}_cls{cls_tag}_spec{spec_tag}_guid{guid_tag}.pt"
    return os.path.join(ckpt_dir, filename)


def train_ddpm_for_ablation(X_train, y_train, variant_config, model_save_path, device='cuda'):
    print(f"  DDPM模型不存在，开始针对变体训练新模型并保存到: {model_save_path}")

    target_psd = compute_target_psd(X_train).to(device)
    target_laterality = compute_class_laterality(X_train, y_train).to(device)

    eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)

    # 训练分类器
    print("  训练分类器...")
    classifier = pretrain_classifier(
        classifier, 
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        epochs=300, batch_size=min(64, len(X_train)), lr=1e-3, device=device, 
        save_path=None, verbose=False
    )

    model = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=CHANNELS,
        n_samples=N_SAMPLES,
        fs=FS,
        c3_idx=C3_IDX,
        c4_idx=C4_IDX
    ).to(device)

    config = variant_config['config']
    epochs = 300
    batch_size = min(32, len(X_train))

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimizer = torch.optim.AdamW(model.eps_model.parameters(), lr=1e-4, weight_decay=0.01)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss, _ = model.loss(
                x_batch, y_batch,
                noise_weight=config.get('noise_weight', 1.0),
                spectral_weight=config.get('spectral_weight', 0.5),
                erd_weight=config.get('erd_weight', 0.5),
                cls_weight=config.get('cls_weight', 1.0)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()

    ckpt = {
        'model_state_dict': model.state_dict(),
        'target_psd': target_psd.detach().cpu(),
        'target_laterality': target_laterality.detach().cpu(),
        'config': config,
    }
    torch.save(ckpt, model_save_path)
    print("  ✅ 新DDPM模型训练完成并已保存")
    return model


def load_pretrained_ddpm(X_train, y_train, variant_config, device='cuda', fold_idx=-1):
    ckpt_path = get_ablation_ckpt_path(variant_config, fold_idx=fold_idx)

    if os.path.exists(ckpt_path):
        print(f"  加载该变体已存在的DDPM模型: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
        classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)

        if isinstance(checkpoint, dict) and 'target_psd' in checkpoint and 'target_laterality' in checkpoint:
            target_psd = checkpoint['target_psd'].to(device)
            target_laterality = checkpoint['target_laterality'].to(device)
        else:
            target_psd = torch.zeros(N_SAMPLES // 2 + 1, device=device)
            target_laterality = torch.zeros(NUM_CLASSES, device=device)

        model = ClassDiscriminativeDDPM(
            eps_model=eps_model,
            classifier=classifier,
            target_psd=target_psd,
            target_laterality=target_laterality,
            n_timesteps=1000,
            channels=CHANNELS,
            n_samples=N_SAMPLES,
            fs=FS,
            c3_idx=C3_IDX,
            c4_idx=C4_IDX
        ).to(device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except RuntimeError as e:
                if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
                    print("  ⚠️  checkpoint参数不完全匹配，使用strict=False加载...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    raise
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("  ✅ 已加载该消融变体专属的DDPM模型")
        return model

    return train_ddpm_for_ablation(X_train, y_train, variant_config, ckpt_path, device=device)


def train_and_eval_classifier(X_train, y_train, X_test, y_test, use_val=False):
    """训练分类器并评估"""
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to('cuda')
    classifier = pretrain_classifier(
        classifier,
        torch.FloatTensor(X_train).to('cuda'),
        torch.LongTensor(y_train).to('cuda'),
        epochs=CLASSIFIER_EPOCHS,
        batch_size=min(CLASSIFIER_BATCH_SIZE, len(X_train)),
        lr=CLASSIFIER_LR,
        device='cuda',
        save_path=None,
        verbose=False
    )
    classifier.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to('cuda')
        pred = classifier(test_tensor).argmax(dim=1).cpu().numpy()
    acc = np.mean(pred == y_test)
    return acc


def evaluate_variant_fold(X_train, y_train, X_test, y_test, variant_config, model, device='cuda'):
    """评估一个变体在一个fold上的表现"""
    guidance_scale = variant_config['config'].get('guidance_scale', 3.0)
    
    # 生成增强数据
    model.eval()
    samples_per_class = max(10, int(len(X_train) // NUM_CLASSES * AUG_RATIO))
    
    X_gen_list = []
    y_gen_list = []
    
    for cls in range(NUM_CLASSES):
        y_tensor = torch.full((samples_per_class,), cls, dtype=torch.long, device=device)
        
        with torch.no_grad():
            samples = model.sample(
                batch_size=samples_per_class,
                y=y_tensor,
                guidance_scale=guidance_scale,
                device=device
            )
        
        X_gen_list.append(samples.cpu().numpy())
        y_gen_list.extend([cls] * samples_per_class)
    
    X_gen = np.concatenate(X_gen_list, axis=0)
    y_gen = np.array(y_gen_list)
    
    X_gen = normalize_generated_data_to_real_stats(X_train, X_gen)

    X_combined = np.concatenate([X_train, X_gen], axis=0)
    y_combined = np.concatenate([y_train, y_gen], axis=0)
    
    # 用增强数据训练分类器
    aug_acc = train_and_eval_classifier(X_combined, y_combined, X_test, y_test)
    
    return aug_acc


def main():
    print("="*60)
    print("消融实验 - PhysioNet MI4C 数据集 - 测试每个组件的贡献")
    print("使用LMSO 10-Fold Cross-Subject方式")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载PhysioNet数据
    print("\n加载PhysioNet MI4C数据...")
    X_all, y_all, subjects, sessions = load_physionet_data()
    
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    print(f"总共有 {n_subjects} 个被试")
    print(f"数据形状: {X_all.shape}")
    print(f"类别分布: {np.bincount(y_all)}")
    
    # 定义变体
    variants = [
        {
            'name': '完整模型 (Full Model)',
            'config': {
                'erd_weight': 2.0,
                'cls_weight': 1.0,
                'noise_weight': 1.0,
                'spectral_weight': 1.0,
                'guidance_scale': 5.2
            }
        },
        {
            'name': '无ERD约束 (w/o ERD)',
            'config': {
                'erd_weight': 0.0,
                'cls_weight': 1.0,
                'noise_weight': 1.0,
                'spectral_weight': 1.0,
                'guidance_scale': 5.2
            }
        },
        {
            'name': '无分类器引导 (w/o Classifier)',
            'config': {
                'erd_weight': 2.0,
                'cls_weight': 0.0,
                'noise_weight': 1.0,
                'spectral_weight': 1.0,
                'guidance_scale': 0.0
            }
        },
        {
            'name': '无频谱约束 (w/o Spectral)',
            'config': {
                'erd_weight': 2.0,
                'cls_weight': 1.0,
                'noise_weight': 1.0,
                'spectral_weight': 0.0,
                'guidance_scale': 5.2
            }
        },
        {
            'name': '仅基础DDPM (Only Noise)',
            'config': {
                'erd_weight': 0.0,
                'cls_weight': 0.0,
                'noise_weight': 1.0,
                'spectral_weight': 0.0,
                'guidance_scale': 0.0
            }
        }
    ]
    
    # 构建LMSO 10-fold
    np.random.seed(42)
    shuffled_ids = np.random.permutation(unique_subjects).tolist()
    fold_size = n_subjects // N_FOLDS
    remainder = n_subjects % N_FOLDS
    folds = []
    start = 0
    for i in range(N_FOLDS):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(shuffled_ids[start:start + size])
        start += size
    
    # 存储所有fold的结果
    all_fold_results = {variant['name']: {'baseline': [], 'augmented': [], 'improvement': []} 
                       for variant in variants}
    
    print("\n" + "="*60)
    print(f"开始LMSO {N_FOLDS}-Fold Cross-Subject消融实验")
    print("="*60)
    
    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        test_subjects = folds[fold_idx]
        train_subjects = [s for s in unique_subjects if s not in test_subjects]
        
        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, test_subjects)
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Test subjects: {len(test_subjects)}")
        
        # Baseline: 不用增强数据
        baseline_acc = train_and_eval_classifier(X_train, y_train, X_test, y_test)
        print(f"  Baseline准确率: {baseline_acc:.4f}")
        
        for variant in variants:
            print(f"\n  --- 变体: {variant['name']} ---")
            try:
                # 加载/训练DDPM模型（使用全部训练数据）
                model = load_pretrained_ddpm(X_train, y_train, variant, device=device, fold_idx=fold_idx)
                
                aug_acc = evaluate_variant_fold(X_train, y_train, X_test, y_test, variant, model, device)
                improvement = aug_acc - baseline_acc
                
                print(f"  增强后准确率: {aug_acc:.4f}, 改进: {improvement:+.4f}")
                
                all_fold_results[variant['name']]['baseline'].append(float(baseline_acc))
                all_fold_results[variant['name']]['augmented'].append(float(aug_acc))
                all_fold_results[variant['name']]['improvement'].append(float(improvement))
            except Exception as e:
                print(f"  错误 ({variant['name']}): {e}")
                import traceback
                traceback.print_exc()
                all_fold_results[variant['name']]['baseline'].append(0.0)
                all_fold_results[variant['name']]['augmented'].append(0.0)
                all_fold_results[variant['name']]['improvement'].append(0.0)
    
    # 计算所有fold的平均结果
    print("\n" + "="*60)
    print("计算所有fold的平均结果")
    print("="*60)
    
    results = []
    for variant in variants:
        variant_name = variant['name']
        baseline_accs = all_fold_results[variant_name]['baseline']
        augmented_accs = all_fold_results[variant_name]['augmented']
        improvements = all_fold_results[variant_name]['improvement']
        
        mean_baseline = np.mean(baseline_accs)
        std_baseline = np.std(baseline_accs)
        mean_augmented = np.mean(augmented_accs)
        std_augmented = np.std(augmented_accs)
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        results.append({
            'variant': variant_name,
            'config': variant['config'],
            'baseline_acc': float(mean_baseline),
            'baseline_std': float(std_baseline),
            'augmented_acc': float(mean_augmented),
            'augmented_std': float(std_augmented),
            'improvement': float(mean_improvement),
            'improvement_std': float(std_improvement),
            'per_fold': {
                'baseline': baseline_accs,
                'augmented': augmented_accs,
                'improvement': improvements
            }
        })
    
    # 保存结果
    output_file = 'outputs/results/paper_experiments/ablation_study_physionet.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print(f"消融实验结果总结 - PhysioNet MI4C（LMSO {N_FOLDS}-Fold Cross-Subject）")
    print("="*60)
    print(f"{'变体':<30} {'Baseline (mean±std)':<20} {'增强后 (mean±std)':<20} {'改进 (mean±std)':<20}")
    print("-"*90)
    for r in results:
        baseline_str = f"{r['baseline_acc']:.4f}±{r['baseline_std']:.4f}"
        augmented_str = f"{r['augmented_acc']:.4f}±{r['augmented_std']:.4f}"
        improvement_str = f"{r['improvement']:+.4f}±{r['improvement_std']:.4f}"
        print(f"{r['variant']:<30} {baseline_str:<20} {augmented_str:<20} {improvement_str:<20}")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
