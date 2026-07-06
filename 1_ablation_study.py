"""
消融实验 (Ablation Study)
测试每个组件的贡献：ERD约束、分类器引导、频谱约束
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
import json
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from core.models.ddpm.class_discriminative import (
    ClassDiscriminativeDDPM, MultiScaleCondUNet, EEGClassifier, pretrain_classifier
)
from utils.data_loader import load_bci2a_data, get_subject_data, get_subject_session_data

# 与敏感度分析 / 全场景评估保持一致的分类器训练超参数
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3

def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd

def compute_class_laterality(X, y, num_classes=4, fs=250, c3_idx=7, c4_idx=11):
    """计算每个类别的平均侧化指数"""
    laterality = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        cls_data = X[y == cls]
        if len(cls_data) == 0:
            laterality[cls] = 0.0
            continue
            
        lat_values = []
        for i in range(len(cls_data)):
            f, psd_c3 = signal.welch(cls_data[i, c3_idx], fs=fs, nperseg=256)
            f, psd_c4 = signal.welch(cls_data[i, c4_idx], fs=fs, nperseg=256)
            alpha_mask = (f >= 8) & (f <= 13)
            c3_alpha = psd_c3[alpha_mask].mean()
            c4_alpha = psd_c4[alpha_mask].mean()
            lat = (c4_alpha - c3_alpha) / (c4_alpha + c3_alpha + 1e-10)
            lat_values.append(lat)
        
        laterality[cls] = float(np.mean(lat_values)) if lat_values else 0.0
    
    return laterality

def normalize_generated_data_to_real_stats(X_real: np.ndarray, X_gen: np.ndarray) -> np.ndarray:
    """
    将生成数据的统计特性对齐到真实数据（与 utils/data_loader.py 一致的逐通道标准化空间）。

    data_loader 中标准化方式为：
        X = (X - mean(axis=(0, 2))) / std(axis=(0, 2))

    这里对齐逻辑为：
        先把生成数据按自身逐通道 mean/std 标准化到近似 z-score 空间，
        再映射到当前训练集的逐通道 mean/std 空间。
    """
    X_real = X_real.astype(np.float32)
    X_gen = X_gen.astype(np.float32)

    real_mean = X_real.mean(axis=(0, 2), keepdims=True)
    real_std = X_real.std(axis=(0, 2), keepdims=True)
    gen_mean = X_gen.mean(axis=(0, 2), keepdims=True)
    gen_std = X_gen.std(axis=(0, 2), keepdims=True)

    eps = 1e-8
    if float(np.max(gen_std)) < eps:
        print("⚠️  生成数据逐通道标准差过小，跳过对齐")
        return X_gen

    X_gen_norm = (X_gen - gen_mean) / (gen_std + eps)
    X_gen_aligned = X_gen_norm * (real_std + eps) + real_mean
    return X_gen_aligned

def load_or_train_classifier(X_train, y_train, device='cuda', classifier_path='checkpoints/classifier_class_disc.pt'):
    """
    加载预训练分类器，如果不存在则训练并保存
    
    注意：默认使用 train_class_discriminative_ddpm.py 中已训练的分类器
    路径：checkpoints/classifier_class_disc.pt
    
    Args:
        X_train: 训练数据（仅在需要重新训练时使用）
        y_train: 训练标签（仅在需要重新训练时使用）
        device: 设备
        classifier_path: 分类器保存路径
    
    Returns:
        classifier: 训练好的分类器
    """
    classifier = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(device)
    
    # 检查是否存在预训练分类器
    if os.path.exists(classifier_path):
        print(f"  加载预训练分类器: {classifier_path}")
        try:
            checkpoint = torch.load(classifier_path, map_location=device)
            # 支持两种保存格式：
            # 1. 包含 'model_state_dict' 的字典格式（我们的统一格式）
            # 2. 直接是 state_dict 的格式（pretrain_classifier 直接保存的格式）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接是 state_dict 格式
                classifier.load_state_dict(checkpoint)
            classifier.eval()
            print("  ✅ 分类器加载成功！")
            return classifier
        except Exception as e:
            print(f"  ⚠️  加载失败 ({e})，将重新训练...")
    
    # 如果不存在或加载失败，则训练
    print("  预训练分类器不存在，开始训练...")
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
    
    # 先训练（不保存，因为我们要保存为统一格式）
    classifier = pretrain_classifier(
        classifier, 
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        epochs=300, batch_size=64, lr=1e-3, device=device, 
        save_path=None, verbose=False  # 不在这里保存，我们自己保存
    )
    
    # 保存为统一格式（包含 model_state_dict 的字典）
    checkpoint_dict = {
        'model_state_dict': classifier.state_dict(),
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
    }
    torch.save(checkpoint_dict, classifier_path)
    print(f"  ✅ 分类器训练完成并已保存到: {classifier_path}")
    return classifier

def _float_to_tag(v: float) -> str:
    """将浮点数转换成适合文件名的字符串，例如 2.0 -> '2p0'。"""
    return str(v).replace('.', 'p')


def get_ablation_ckpt_path(variant_config: dict) -> str:
    """
    为每个消融变体生成独立的checkpoint路径。
    命名中编码 erd_weight / cls_weight / spectral_weight / guidance_scale。
    """
    cfg = variant_config['config']
    erd_tag = _float_to_tag(cfg.get('erd_weight', 0.0))
    cls_tag = _float_to_tag(cfg.get('cls_weight', 0.0))
    spec_tag = _float_to_tag(cfg.get('spectral_weight', 0.0))
    guid_tag = _float_to_tag(cfg.get('guidance_scale', 0.0))

    ckpt_dir = os.path.join('checkpoints', 'ablation_models')
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"ddpm_erd{erd_tag}_cls{cls_tag}_spec{spec_tag}_guid{guid_tag}.pt"
    return os.path.join(ckpt_dir, filename)


def train_ddpm_for_ablation(X_train, y_train, variant_config, model_save_path, device='cuda'):
    """
    针对单个消融变体，从零开始训练一个DDPM，并保存到独立checkpoint。
    训练时使用该变体指定的 erd_weight / cls_weight / spectral_weight 配置。
    """
    print(f"  DDPM模型不存在，开始针对变体训练新模型并保存到: {model_save_path}")

    # 计算该被试训练集的目标统计量
    target_psd = compute_target_psd(X_train).to(device)
    target_laterality = compute_class_laterality(X_train, y_train).to(device)

    # 创建噪声预测网络
    eps_model = MultiScaleCondUNet(channels=22, num_classes=4).to(device)

    # 加载/训练分类器（用于classifier guidance）
    classifier = load_or_train_classifier(X_train, y_train, device=device)

    # 构建DDPM
    model = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=22,
        n_samples=1000,
        fs=250
    ).to(device)

    # 训练配置（与原脚本保持一致）
    config = variant_config['config']
    epochs = 300
    batch_size = 32

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

    # 保存checkpoint，包含模型权重和目标统计量，便于后续重复使用
    ckpt = {
        'model_state_dict': model.state_dict(),
        'target_psd': target_psd.detach().cpu(),
        'target_laterality': target_laterality.detach().cpu(),
        'config': config,
    }
    torch.save(ckpt, model_save_path)
    print("  ✅ 新DDPM模型训练完成并已保存")
    return model


def load_pretrained_ddpm(X_train, y_train, variant_config, device='cuda'):
    """
    为【指定消融变体】加载或训练对应的DDPM模型。

    - 每个变体都有独立的 checkpoint 路径（通过其损失权重+guidance 配置编码）
    - 如果该checkpoint已存在，直接加载；
    - 如果不存在，则调用 train_ddpm_for_ablation 先训练再缓存。
    """
    ckpt_path = get_ablation_ckpt_path(variant_config)

    if os.path.exists(ckpt_path):
        print(f"  加载该变体已存在的DDPM模型: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        eps_model = MultiScaleCondUNet(channels=22, num_classes=4).to(device)
        classifier = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(device)

        # 从checkpoint中读取目标统计量，若不存在则回退为零向量
        if isinstance(checkpoint, dict) and 'target_psd' in checkpoint and 'target_laterality' in checkpoint:
            target_psd = checkpoint['target_psd'].to(device)
            target_laterality = checkpoint['target_laterality'].to(device)
        else:
            target_psd = torch.zeros(501, device=device)
            target_laterality = torch.zeros(4, device=device)

        model = ClassDiscriminativeDDPM(
            eps_model=eps_model,
            classifier=classifier,
            target_psd=target_psd,
            target_laterality=target_laterality,
            n_timesteps=1000,
            channels=22,
            n_samples=1000,
            fs=250
        ).to(device)

        # 加载权重（优先使用model_state_dict，兼容旧格式）
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

    # 如果不存在，则针对该变体重新训练一个DDPM
    return train_ddpm_for_ablation(X_train, y_train, variant_config, ckpt_path, device=device)

def load_baseline_result(baseline_path='outputs/results/baseline_all_scenarios.json'):
    """
    尝试从已有结果文件加载baseline准确率
    
    Args:
        baseline_path: baseline结果文件路径
    
    Returns:
        baseline_acc: baseline准确率，如果文件不存在则返回None
    """
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            # 尝试从不同场景中获取baseline结果
            # 优先使用cross_session场景的结果
            if 'cross_session' in baseline_data:
                baseline_acc = baseline_data['cross_session'].get('mean', None)
                if baseline_acc is not None:
                    print(f"  从 {baseline_path} 加载baseline结果: {baseline_acc:.4f}")
                    return baseline_acc
            # 如果没有cross_session，尝试其他场景
            for key in ['within_subject', 'cross_subject', 'mean']:
                if key in baseline_data:
                    baseline_acc = baseline_data[key].get('mean', baseline_data[key]) if isinstance(baseline_data[key], dict) else baseline_data[key]
                    if baseline_acc is not None:
                        print(f"  从 {baseline_path} 加载baseline结果: {baseline_acc:.4f}")
                        return baseline_acc
        except Exception as e:
            print(f"  读取baseline结果失败: {e}")
    return None

def evaluate_variant(X_train, y_train, X_test, y_test, variant_config, device='cuda', baseline_acc_precomputed=None):
    """评估一个变体"""
    print(f"\n{'='*60}")
    print(f"测试变体: {variant_config['name']}")
    print(f"配置: {variant_config['config']}")
    print(f"{'='*60}")
    
    # 针对当前变体加载/训练独立的DDPM模型
    print("加载/训练该变体对应的DDPM模型...")
    model = load_pretrained_ddpm(X_train, y_train, variant_config, device=device)
    
    # 从模型中获取分类器（用于baseline评估）
    classifier = model.classifier
    
    # 生成增强数据
    print("生成增强数据...")
    model.eval()
    n_augment = len(X_train)
    guidance_scale = variant_config['config'].get('guidance_scale', 3.0)
    
    X_gen_list = []
    y_gen_list = []
    
    # 按类别生成
    unique_classes = np.unique(y_train)
    for cls in unique_classes:
        n_class = np.sum(y_train == cls)
        y_tensor = torch.full((n_class,), int(cls), dtype=torch.long, device=device)
        
        with torch.no_grad():
            samples = model.sample(
                batch_size=n_class,
                y=y_tensor,
                guidance_scale=guidance_scale,
                device=device
            )
        
        X_gen_list.append(samples.cpu().numpy())
        y_gen_list.extend([cls] * n_class)
    
    X_gen = np.concatenate(X_gen_list, axis=0)
    y_gen = np.array(y_gen_list)
    
    # 将生成数据对齐到真实训练数据的逐通道统计空间（与敏感度分析/评估脚本一致）
    X_gen = normalize_generated_data_to_real_stats(X_train, X_gen)

    # 合并数据：原始训练数据 + 生成数据
    X_combined = np.concatenate([X_train, X_gen], axis=0)
    y_combined = np.concatenate([y_train, y_gen], axis=0)
    print(f"  合并后训练集: {X_combined.shape} (原始: {X_train.shape}, 生成: {X_gen.shape})")
    
    # 评估1: Baseline - 使用预训练分类器在测试集上直接评估（不重新训练）
    if baseline_acc_precomputed is not None:
        print(f"使用预计算的Baseline准确率: {baseline_acc_precomputed:.4f}")
        baseline_acc = baseline_acc_precomputed
    else:
        print("评估Baseline（使用预训练分类器在测试集上评估）...")
        classifier.eval()  # 使用已经加载的分类器
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(device)
            baseline_pred = classifier(test_tensor).argmax(dim=1).cpu().numpy()
        baseline_acc = np.mean(baseline_pred == y_test)
    
    # 评估2: Augmented - 用原始训练数据+生成数据重新训练EEGClassifier，在测试集上评估
    print("评估增强方法（原始训练数据+生成数据重新训练分类器）...")
    augmented_classifier = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(device)
    # 用原始+生成数据重新训练分类器（与敏感度分析/评估脚本一致）
    augmented_classifier = pretrain_classifier(
        augmented_classifier,
        torch.FloatTensor(X_combined).to(device),
        torch.LongTensor(y_combined).to(device),
        epochs=CLASSIFIER_EPOCHS,
        batch_size=CLASSIFIER_BATCH_SIZE,
        lr=CLASSIFIER_LR,
        device=device,
        verbose=False
    )
    augmented_classifier.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(device)
        augmented_pred = augmented_classifier(test_tensor).argmax(dim=1).cpu().numpy()
    augmented_acc = np.mean(augmented_pred == y_test)
    improvement = augmented_acc - baseline_acc
    
    print(f"\nBaseline准确率: {baseline_acc:.4f}")
    print(f"增强后准确率: {augmented_acc:.4f}")
    print(f"改进: {improvement:+.4f}")
    
    return {
        'variant': variant_config['name'],
        'config': variant_config['config'],
        'baseline_acc': float(baseline_acc),
        'augmented_acc': float(augmented_acc),
        'improvement': float(improvement)
    }

def main():
    print("="*60)
    print("消融实验 - 测试每个组件的贡献")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据（跨会话设置：训练集=Session 1，测试集=Session 2）
    print("\n加载数据（跨会话设置 - 所有9个被试）...")
    X_all, y_all, subjects, sessions = load_bci2a_data()
    
    n_subjects = len(np.unique(subjects))
    print(f"总共有 {n_subjects} 个被试")
    
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
                'erd_weight': 0.0,  # 与完整模型保持相同的guidance scale，仅移除ERD约束
                'cls_weight': 1.0,
                'noise_weight': 1.0,
                'spectral_weight': 1.0,
                'guidance_scale': 5.2  # 与完整模型相同，公平对比
            }
        },
        {
            'name': '无分类器引导 (w/o Classifier)',
            'config': {
                'erd_weight':2.0,
                'cls_weight': 0.0,
                'noise_weight': 1.0,
                'spectral_weight': 1.0,
                'guidance_scale': 0.0  # 无分类器引导
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
    
    # 尝试加载已有的baseline结果
    print("\n尝试加载已有的baseline结果...")
    baseline_acc_precomputed = load_baseline_result()
    
    # 存储所有被试的结果
    all_subject_results = {variant['name']: {'baseline': [], 'augmented': [], 'improvement': []} 
                          for variant in variants}
    
    # 对每个被试进行跨会话实验
    print("\n" + "="*60)
    print("开始跨会话消融实验（所有9个被试）")
    print("="*60)
    
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"被试 {subject_id+1}/{n_subjects}")
        print(f"{'='*60}")
        
        # 获取该被试的跨会话数据
        X_train, y_train = get_subject_session_data(X_all, y_all, subjects, sessions, subject_id, session_id=0)
        X_test, y_test = get_subject_session_data(X_all, y_all, subjects, sessions, subject_id, session_id=1)
        
        print(f"  训练集 (Session 1): {X_train.shape}, 类别分布: {np.bincount(y_train)}")
        print(f"  测试集 (Session 2): {X_test.shape}, 类别分布: {np.bincount(y_test)}")
        
        # 对每个变体进行评估
        for variant in variants:
            try:
                result = evaluate_variant(X_train, y_train, X_test, y_test, variant, device, baseline_acc_precomputed)
                all_subject_results[variant['name']]['baseline'].append(result['baseline_acc'])
                all_subject_results[variant['name']]['augmented'].append(result['augmented_acc'])
                all_subject_results[variant['name']]['improvement'].append(result['improvement'])
            except Exception as e:
                print(f"  错误 ({variant['name']}): {e}")
                all_subject_results[variant['name']]['baseline'].append(0.0)
                all_subject_results[variant['name']]['augmented'].append(0.0)
                all_subject_results[variant['name']]['improvement'].append(0.0)
    
    # 计算所有被试的平均结果
    print("\n" + "="*60)
    print("计算所有被试的平均结果")
    print("="*60)
    
    results = []
    for variant in variants:
        variant_name = variant['name']
        baseline_accs = all_subject_results[variant_name]['baseline']
        augmented_accs = all_subject_results[variant_name]['augmented']
        improvements = all_subject_results[variant_name]['improvement']
        
        # 计算平均值和标准差
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
            'per_subject': {
                'baseline': baseline_accs,
                'augmented': augmented_accs,
                'improvement': improvements
            }
        })
    
    # 保存结果
    output_file = 'outputs/results/paper_experiments/ablation_study.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("消融实验结果总结（跨会话，9个被试平均）")
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
