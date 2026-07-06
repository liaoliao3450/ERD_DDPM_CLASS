#!/usr/bin/env python3
"""
Class-Discriminative DDPM 方法评估（数据增强）

数据处理方式与 evaluate_EEGNET_baseline.py 完全一致：
  - 数据加载: load_dataset_data(dataset) (BCI2a/BCI2b/PhysioNet)
  - 标准化: 按通道 z-score
  - 测试场景: Within-Subject / Cross-Session / Cross-Subject (LOSO 或 LMSO 10-Fold)
  - 评估指标: Accuracy + Cohen's Kappa

唯一区别：训练分类器时，在真实训练集上加入 DDPM 生成的等量增强样本。

Usage:
    python experiments/paper_experiments/evaluate_Class-Discriminativ_ddpm.py
    python experiments/paper_experiments/evaluate_Class-Discriminativ_ddpm.py --dataset bci2b
    python experiments/paper_experiments/evaluate_Class-Discriminativ_ddpm.py --dataset physionet
"""
import sys, os, argparse, json, torch, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')
sys.path.insert(0, 'experiments/paper_experiments')
from class_discriminative import (
    EEGClassifier, pretrain_classifier,
    MultiScaleCondUNet, ClassDiscriminativeDDPM
)
from config import DATASETS, get_dataset_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子，保证结果可复现
SEED = 42
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DDPM 模型加载
# ============================================================================
# 各数据集的 checkpoint 路径
CHECKPOINT_PATHS = {
    'bci2a': 'checkpoints/best_class_discriminative.pt',
    'bci2b': 'checkpoints/best_class_discriminative_bci2b.pt',
    'physionet': 'checkpoints/best_class_discriminative_physionet.pt',
}


def load_ddpm(dataset, cfg, device='cuda'):
    """加载指定数据集的 Class-Discriminative DDPM 模型"""
    checkpoint_path = CHECKPOINT_PATHS.get(dataset)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print(f"请先为 {dataset} 训练 DDPM 模型")
        return None, False

    print(f"📥 加载 DDPM 模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    channels = cfg['channels']
    n_samples = cfg['n_samples']
    num_classes = cfg['num_classes']
    fs = cfg['fs']

    eps_model = MultiScaleCondUNet(channels=channels, num_classes=num_classes).to(device)
    classifier = EEGClassifier(channels=channels, n_samples=n_samples, num_classes=num_classes).to(device)

    # 从 checkpoint 读取 target_psd / target_laterality
    if isinstance(checkpoint, dict) and 'target_psd' in checkpoint and 'target_laterality' in checkpoint:
        target_psd = checkpoint['target_psd'].to(device)
        target_laterality = checkpoint['target_laterality'].to(device)
    else:
        # 兜底：用零向量
        target_psd = torch.zeros(n_samples // 2 + 1).to(device)
        target_laterality = torch.zeros(num_classes).to(device)

    ddpm = ClassDiscriminativeDDPM(
        eps_model, classifier,
        target_psd, target_laterality,
        n_timesteps=1000, channels=channels, n_samples=n_samples, fs=fs
    ).to(device)

    # 加载 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        try:
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError:
            print("  警告: checkpoint 参数不完全匹配，使用 strict=False 加载")
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        ddpm.load_state_dict(checkpoint)

    ddpm.eval()
    print("✅ DDPM 模型加载成功")
    return ddpm, True


# ============================================================================
# 数据加载 (与 baseline 完全一致)
# ============================================================================
def load_dataset_data(dataset='bci2a'):
    """加载指定数据集的数据（与 evaluate_EEGNET_baseline.py 完全一致）"""
    cfg = get_dataset_config(dataset)
    num_classes = cfg['num_classes']

    print(f"加载 {dataset} 数据...")

    if dataset == 'bci2a':
        from data_loader import load_bci2a_data
        X, y, subjects, sessions = load_bci2a_data()
    elif dataset == 'bci2b':
        from data_loader_bci2b import load_bci2b_data
        X, y, subjects, sessions, _ = load_bci2b_data()
    elif dataset == 'physionet':
        from data_loader_physionet_mi4c import load_physionet_mi4c_data
        X, y, subjects, sessions, _ = load_physionet_mi4c_data()
    else:
        raise ValueError(f"未知数据集: {dataset}")

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # 标签从 0 开始 + 过滤掉超过 num_classes 的类别
    y = y - y.min()
    mask = y < num_classes
    X, y = X[mask], y[mask]
    subjects = np.asarray(subjects)[mask]
    sessions = np.asarray(sessions)[mask]

    # 按通道 z-score 标准化（与各 data_loader 保持一致）
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-8)

    print(f"  数据形状: {X.shape}")
    print(f"  类别分布: {np.bincount(y)}")
    print(f"  被试数: {len(np.unique(subjects))}, 会话数: {len(np.unique(sessions))}")
    print(f"  标准化后范围: [{X.min():.4f}, {X.max():.4f}], mean={X.mean():.4f}, std={X.std():.4f}")

    return X, y, subjects, sessions


# ============================================================================
# 生成样本 + 对齐
# ============================================================================
def generate_samples(ddpm, n_per_class, num_classes, guidance_scale=5.5):
    """用 DDPM 生成样本（DDIM 采样）"""
    ddpm.eval()
    gen_X, gen_y = [], []

    with torch.no_grad():
        for c in range(num_classes):
            n_batches = (n_per_class + 49) // 50
            generated = 0
            for _ in range(n_batches):
                batch_size = min(50, n_per_class - generated)
                if batch_size <= 0:
                    break
                yg = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample_ddim(batch_size, yg, steps=50,
                                           guidance_scale=guidance_scale, device=DEVICE)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)
                generated += batch_size

    return np.concatenate(gen_X), np.array(gen_y)


def align_to_real_stats(X_real, X_gen):
    """将生成数据按通道对齐到真实训练数据的统计空间

    与 data_loader 的逐通道 z-score 标准化空间一致：
      先把生成数据按自身逐通道 mean/std 标准化，
      再映射到真实训练数据的逐通道 mean/std 空间。
    """
    real_mean = X_real.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    real_std = X_real.std(axis=(0, 2), keepdims=True)
    gen_mean = X_gen.mean(axis=(0, 2), keepdims=True)
    gen_std = X_gen.std(axis=(0, 2), keepdims=True)

    eps = 1e-8
    if float(np.max(gen_std)) < eps:
        print("⚠️  生成数据逐通道标准差过小，跳过对齐")
        return X_gen.astype(np.float32)

    X_gen_norm = (X_gen - gen_mean) / (gen_std + eps)
    X_gen_aligned = X_gen_norm * (real_std + eps) + real_mean
    return X_gen_aligned.astype(np.float32)


# ============================================================================
# 评估函数 (与 baseline 结构一致，训练时加入 DDPM 增强)
# ============================================================================
def within_subject_test(ddpm, X, y, subjects, cfg):
    """被试内测试（DDPM 增强）"""
    print("\n" + "="*70)
    print("1. Within-Subject 测试 (DDPM 增强)")
    print("="*70)

    results = []
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    channels = cfg['channels']
    n_samples = cfg['n_samples']
    num_classes = cfg['num_classes']

    for subj_id in unique_subjects:
        print(f"\n被试 {subj_id+1}/{n_subjects}:")

        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
        )

        # 生成与训练集等量的增强样本
        samples_per_class = max(1, len(X_train) // num_classes)
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = generate_samples(ddpm, samples_per_class, num_classes)
        gen_X = align_to_real_stats(X_train, gen_X)

        # 合并真实 + 增强
        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        set_seed()
        clf = EEGClassifier(channels=channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=cfg['classifier_epochs'], batch_size=cfg['classifier_batch_size'],
            lr=cfg['classifier_lr'], device=DEVICE, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'acc': acc, 'kappa': kappa})
        print(f"  准确率: {acc*100:.2f}%, Kappa: {kappa:.4f}")

    if not results:
        return {'mean': 0.0, 'std': 0.0, 'per_subject': [], 'skipped': True,
                'mean_kappa': 0.0, 'std_kappa': 0.0, 'per_subject_kappa': []}

    accs = [r['acc'] for r in results]
    kappas = [r['kappa'] for r in results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)
    print(f"\n平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"平均 Kappa:   {mean_kappa:.4f} ± {std_kappa:.4f}")

    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(a) for a in accs],
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'per_subject_kappa': [float(k) for k in kappas],
    }


def cross_session_test(ddpm, X, y, subjects, sessions, cfg):
    """跨会话测试（DDPM 增强）"""
    print("\n" + "="*70)
    print("2. Cross-Session 测试 (DDPM 增强)")
    print("="*70)

    results = []
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    n_sessions = cfg.get('n_sessions', 2)
    channels = cfg['channels']
    n_samples = cfg['n_samples']
    num_classes = cfg['num_classes']

    if n_sessions < 2:
        print(f"  ⚠️  {cfg.get('data_dir', '')} 只有 1 个 session，跳过 Cross-Session 测试")
        return {'mean': 0.0, 'std': 0.0, 'per_subject': [], 'skipped': True,
                'mean_kappa': 0.0, 'std_kappa': 0.0, 'per_subject_kappa': []}

    for subj_id in unique_subjects:
        print(f"\n被试 {subj_id+1}/{n_subjects}:")

        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠️  数据不足，跳过 (train={len(X_train)}, test={len(X_test)})")
            continue

        samples_per_class = max(1, len(X_train) // num_classes)
        print(f"  Session 0训练: {len(X_train)}, Session 1测试: {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = generate_samples(ddpm, samples_per_class, num_classes)
        gen_X = align_to_real_stats(X_train, gen_X)

        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        set_seed()
        clf = EEGClassifier(channels=channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=cfg['classifier_epochs'], batch_size=cfg['classifier_batch_size'],
            lr=cfg['classifier_lr'], device=DEVICE, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'acc': acc, 'kappa': kappa})
        print(f"  准确率: {acc*100:.2f}%, Kappa: {kappa:.4f}")

    if not results:
        return {'mean': 0.0, 'std': 0.0, 'per_subject': [], 'skipped': True,
                'mean_kappa': 0.0, 'std_kappa': 0.0, 'per_subject_kappa': []}

    accs = [r['acc'] for r in results]
    kappas = [r['kappa'] for r in results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)
    print(f"\n平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"平均 Kappa:   {mean_kappa:.4f} ± {std_kappa:.4f}")

    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(a) for a in accs],
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'per_subject_kappa': [float(k) for k in kappas],
    }


def cross_subject_test(ddpm, X, y, subjects, sessions, cfg):
    """跨被试测试（DDPM 增强）

    支持两种模式（由 cfg['cross_subject_mode'] 决定）：
    - 'loso' (默认): Leave-One-Subject-Out
    - 'lmso_10fold': 十折交叉验证
    """
    mode = cfg.get('cross_subject_mode', 'loso')
    print("\n" + "="*70)
    print(f"3. Cross-Subject 测试 (DDPM 增强, {mode.upper()})")
    print("="*70)

    results = []
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    channels = cfg['channels']
    n_samples = cfg['n_samples']
    num_classes = cfg['num_classes']

    # 构造训练/测试的被试分组
    if mode == 'lmso_10fold':
        np.random.seed(42)
        shuffled_ids = np.random.permutation(unique_subjects).tolist()
        n_folds = 10
        fold_size = n_subjects // n_folds
        remainder = n_subjects % n_folds
        folds = []
        start = 0
        for i in range(n_folds):
            size = fold_size + (1 if i < remainder else 0)
            folds.append(shuffled_ids[start:start + size])
            start += size
        print(f"  LMSO 10-Fold: {n_subjects} 个被试分成 {n_folds} 折")
        for i, fold in enumerate(folds):
            print(f"    Fold {i+1}: {len(fold)} 个测试被试")
        iter_list = enumerate(folds)
    else:
        iter_list = enumerate(unique_subjects)

    for fold_idx, test_group in iter_list:
        if mode == 'lmso_10fold':
            test_subjects = test_group
            train_subjects = [s for s in unique_subjects if s not in test_subjects]
            print(f"\n--- Fold {fold_idx+1}/10 (test: {len(test_subjects)} 被试, train: {len(train_subjects)} 被试) ---")
            train_mask = np.isin(subjects, train_subjects)  # 不限制 session，用所有数据
            test_mask = np.isin(subjects, test_subjects)
        else:
            test_subj = test_group
            print(f"\n测试被试 {test_subj+1}/{n_subjects}:")
            train_mask = (subjects != test_subj) & (sessions == 0)
            test_mask = (subjects == test_subj) & (sessions == 0)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  ⚠️  数据不足，跳过 (train={len(X_train)}, test={len(X_test)})")
            continue

        samples_per_class = max(1, len(X_train) // num_classes)
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}, 生成: {samples_per_class}×{num_classes}")

        gen_X, gen_y = generate_samples(ddpm, samples_per_class, num_classes)
        gen_X = align_to_real_stats(X_train, gen_X)

        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])

        set_seed()
        clf = EEGClassifier(channels=channels, n_samples=n_samples, num_classes=num_classes).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=cfg['classifier_epochs'], batch_size=cfg['classifier_batch_size'],
            lr=cfg['classifier_lr'], device=DEVICE, verbose=False
        )

        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, pred)
        kappa = cohen_kappa_score(y_test, pred)
        results.append({'acc': acc, 'kappa': kappa})
        print(f"  准确率: {acc*100:.2f}%, Kappa: {kappa:.4f}")

    if not results:
        return {'mean': 0.0, 'std': 0.0, 'per_subject': [], 'skipped': True,
                'mean_kappa': 0.0, 'std_kappa': 0.0, 'per_subject_kappa': []}

    accs = [r['acc'] for r in results]
    kappas = [r['kappa'] for r in results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)
    print(f"\n平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"平均 Kappa:   {mean_kappa:.4f} ± {std_kappa:.4f}")

    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(a) for a in accs],
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'per_subject_kappa': [float(k) for k in kappas],
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Class-Discriminative DDPM 数据增强评估')
    parser.add_argument('--dataset', type=str, default='bci2a',
                        choices=['bci2a', 'bci2b', 'physionet'],
                        help='数据集名称')
    parser.add_argument('--guidance_scale', type=float, default=5.5,
                        help='DDPM 分类器引导强度')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all', 'within_subject', 'cross_session', 'cross_subject'],
                        help='测试场景 (默认 all)')
    args = parser.parse_args()

    print("="*70)
    print(f"📊 Class-Discriminative DDPM 数据增强评估 - {args.dataset.upper()}")
    print("="*70)
    print(f"设备: {DEVICE}")
    print(f"guidance_scale: {args.guidance_scale}\n")

    # 获取数据集配置
    cfg = get_dataset_config(args.dataset)

    # 加载 DDPM 模型
    ddpm, loaded = load_ddpm(args.dataset, cfg, DEVICE)
    if not loaded:
        print(f"❌ 未找到 {args.dataset} 的 DDPM 模型！")
        print(f"请先运行: python experiments/paper_experiments/train_class_discriminative_ddpm.py --dataset {args.dataset}")
        return

    # 加载数据（与 baseline 完全一致）
    X, y, subjects, sessions = load_dataset_data(args.dataset)

    # 根据 scenario 参数决定跑哪些测试
    results = {}
    run_all = (args.scenario == 'all')
    skip_ws = args.dataset == 'physionet'
    skip_cs = args.dataset == 'physionet'
    skip_val = {'mean': 0.0, 'std': 0.0, 'per_subject': [], 'skipped': True,
                'mean_kappa': 0.0, 'std_kappa': 0.0, 'per_subject_kappa': []}

    if run_all and skip_ws:
        print("\n⚠️  PhysioNet 跳过 Within-Subject")
        results['within_subject'] = skip_val
    elif run_all or args.scenario == 'within_subject':
        results['within_subject'] = within_subject_test(ddpm, X, y, subjects, cfg)
    else:
        results['within_subject'] = skip_val

    if run_all and skip_cs:
        results['cross_session'] = skip_val
    elif run_all or args.scenario == 'cross_session':
        results['cross_session'] = cross_session_test(ddpm, X, y, subjects, sessions, cfg)
    else:
        results['cross_session'] = skip_val

    if run_all or args.scenario == 'cross_subject':
        results['cross_subject'] = cross_subject_test(ddpm, X, y, subjects, sessions, cfg)
    else:
        results['cross_subject'] = skip_val

    # 汇总
    print("\n" + "="*70)
    print(f"📊 DDPM 数据增强最终结果汇总 - {args.dataset.upper()}")
    print("="*70)

    print(f"\n{'场景':<20} {'准确率':<18} {'Kappa':<18}")
    print("-" * 60)
    for scenario in ['within_subject', 'cross_session', 'cross_subject']:
        r = results[scenario]
        if r.get('skipped'):
            print(f"{scenario:<20} {'跳过':<18}")
        else:
            acc_str = f"{r['mean']*100:.2f}% ± {r['std']*100:.2f}%"
            kappa_str = f"{r['mean_kappa']:.4f} ± {r['std_kappa']:.4f}"
            print(f"{scenario:<20} {acc_str:<18} {kappa_str:<18}")

    # 保存
    os.makedirs('outputs/results', exist_ok=True)

    final_results = {
        'method': 'Class-Discriminative DDPM',
        'dataset': args.dataset,
        'guidance_scale': args.guidance_scale,
        'within_subject': results['within_subject'],
        'cross_session': results['cross_session'],
        'cross_subject': results['cross_subject']
    }

    output_path = f'outputs/results/ddpm_{args.dataset}_all_scenarios.json'
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n💾 结果已保存到: {output_path}")
    print("\n" + "="*70)
    print("✅ DDPM 数据增强评估完成！")
    print("="*70)


if __name__ == '__main__':
    main()
