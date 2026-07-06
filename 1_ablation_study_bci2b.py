"""
消融实验 - BCI2b 数据集
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
from utils.data_loader_bci2b import load_bci2b_data

# BCI2b 数据配置
CHANNELS = 3
N_SAMPLES = 1000
NUM_CLASSES = 2
FS = 250

# 与敏感度分析 / 全场景评估保持一致的分类器训练超参数
CLASSIFIER_EPOCHS = 200
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3

def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd

def compute_class_laterality(X, y, num_classes=NUM_CLASSES, fs=FS, c3_idx=0, c4_idx=2):
    """计算每个类别的平均侧化指数（BCI2b: C3=ch0, C4=ch2）"""
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
    """将生成数据的统计特性对齐到真实数据"""
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

def load_or_train_classifier(X_train, y_train, device='cuda',
                              classifier_path='checkpoints/classifier_class_disc_bci2b.pt'):
    """加载预训练分类器，如果不存在则训练并保存"""
    classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(classifier_path):
        print(f"  加载预训练分类器: {classifier_path}")
        try:
            checkpoint = torch.load(classifier_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            classifier.eval()
            print("  ✅ 分类器加载成功！")
            return classifier
        except Exception as e:
            print(f"  ⚠️  加载失败 ({e})，将重新训练...")
    
    print("  预训练分类器不存在，开始训练...")
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
    
    classifier = pretrain_classifier(
        classifier, 
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        epochs=300, batch_size=64, lr=1e-3, device=device, 
        save_path=None, verbose=False
    )
    
    checkpoint_dict = {
        'model_state_dict': classifier.state_dict(),
        'channels': CHANNELS,
        'n_samples': N_SAMPLES,
        'num_classes': NUM_CLASSES,
    }
    torch.save(checkpoint_dict, classifier_path)
    print(f"  ✅ 分类器训练完成并已保存到: {classifier_path}")
    return classifier

def _float_to_tag(v: float) -> str:
    return str(v).replace('.', 'p')

def get_ablation_ckpt_path(variant_config: dict) -> str:
    cfg = variant_config['config']
    erd_tag = _float_to_tag(cfg.get('erd_weight', 0.0))
    cls_tag = _float_to_tag(cfg.get('cls_weight', 0.0))
    spec_tag = _float_to_tag(cfg.get('spectral_weight', 0.0))
    guid_tag = _float_to_tag(cfg.get('guidance_scale', 0.0))

    ckpt_dir = os.path.join('checkpoints', 'ablation_models_bci2b')
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = f"ddpm_erd{erd_tag}_cls{cls_tag}_spec{spec_tag}_guid{guid_tag}.pt"
    return os.path.join(ckpt_dir, filename)


def train_ddpm_for_ablation(X_train, y_train, variant_config, model_save_path, device='cuda'):
    print(f"  DDPM模型不存在，开始针对变体训练新模型并保存到: {model_save_path}")

    target_psd = compute_target_psd(X_train).to(device)
    target_laterality = compute_class_laterality(X_train, y_train).to(device)

    eps_model = MultiScaleCondUNet(channels=CHANNELS, num_classes=NUM_CLASSES).to(device)
    classifier = load_or_train_classifier(X_train, y_train, device=device)

    model = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=CHANNELS,
        n_samples=N_SAMPLES,
        fs=FS,
        c3_idx=0,
        c4_idx=2
    ).to(device)

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
    ckpt_path = get_ablation_ckpt_path(variant_config)

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
            c3_idx=0,
            c4_idx=2
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

def load_baseline_result(baseline_path='outputs/results/baseline_all_scenarios_bci2b.json'):
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            for key in ['cross_session', 'within_subject', 'cross_subject', 'mean']:
                if key in baseline_data:
                    baseline_acc = baseline_data[key].get('mean', baseline_data[key]) if isinstance(baseline_data[key], dict) else baseline_data[key]
                    if baseline_acc is not None:
                        print(f"  从 {baseline_path} 加载baseline结果: {baseline_acc:.4f}")
                        return baseline_acc
        except Exception as e:
            print(f"  读取baseline结果失败: {e}")
    return None

def evaluate_variant(X_train, y_train, X_test, y_test, variant_config, device='cuda', baseline_acc_precomputed=None):
    print(f"\n{'='*60}")
    print(f"测试变体: {variant_config['name']}")
    print(f"配置: {variant_config['config']}")
    print(f"{'='*60}")
    
    print("加载/训练该变体对应的DDPM模型...")
    model = load_pretrained_ddpm(X_train, y_train, variant_config, device=device)
    
    classifier = model.classifier
    
    print("生成增强数据...")
    model.eval()
    guidance_scale = variant_config['config'].get('guidance_scale', 3.0)
    
    X_gen_list = []
    y_gen_list = []
    
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
    
    X_gen = normalize_generated_data_to_real_stats(X_train, X_gen)

    X_combined = np.concatenate([X_train, X_gen], axis=0)
    y_combined = np.concatenate([y_train, y_gen], axis=0)
    print(f"  合并后训练集: {X_combined.shape} (原始: {X_train.shape}, 生成: {X_gen.shape})")
    
    if baseline_acc_precomputed is not None:
        print(f"使用预计算的Baseline准确率: {baseline_acc_precomputed:.4f}")
        baseline_acc = baseline_acc_precomputed
    else:
        print("评估Baseline...")
        classifier.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(device)
            baseline_pred = classifier(test_tensor).argmax(dim=1).cpu().numpy()
        baseline_acc = np.mean(baseline_pred == y_test)
    
    print("评估增强方法（原始训练数据+生成数据重新训练分类器）...")
    augmented_classifier = EEGClassifier(channels=CHANNELS, n_samples=N_SAMPLES, num_classes=NUM_CLASSES).to(device)
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
    print("消融实验 - BCI2b 数据集 - 测试每个组件的贡献")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载BCI2b数据
    print("\n加载BCI2b数据...")
    X_all, y_all, subjects, sessions, _ = load_bci2b_data()
    
    n_subjects = len(np.unique(subjects))
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
    
    # 尝试加载已有的baseline结果
    print("\n尝试加载已有的baseline结果...")
    baseline_acc_precomputed = load_baseline_result()
    
    # 存储所有被试的结果
    all_subject_results = {variant['name']: {'baseline': [], 'augmented': [], 'improvement': []} 
                          for variant in variants}
    
    # 对每个被试进行跨会话实验
    print("\n" + "="*60)
    print(f"开始跨会话消融实验（所有{n_subjects}个被试）")
    print("="*60)
    
    for subject_id in range(n_subjects):
        print(f"\n{'='*60}")
        print(f"被试 {subject_id+1}/{n_subjects}")
        print(f"{'='*60}")
        
        # 获取该被试的跨会话数据：Session 0 (T) 训练, Session 1 (E) 测试
        train_mask = (subjects == subject_id) & (sessions == 0)
        test_mask = (subjects == subject_id) & (sessions == 1)
        
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  被试 {subject_id} 数据不足，跳过")
            for variant in variants:
                all_subject_results[variant['name']]['baseline'].append(0.0)
                all_subject_results[variant['name']]['augmented'].append(0.0)
                all_subject_results[variant['name']]['improvement'].append(0.0)
            continue
        
        print(f"  训练集 (Session T): {X_train.shape}, 类别分布: {np.bincount(y_train)}")
        print(f"  测试集 (Session E): {X_test.shape}, 类别分布: {np.bincount(y_test)}")
        
        for variant in variants:
            try:
                result = evaluate_variant(X_train, y_train, X_test, y_test, variant, device, baseline_acc_precomputed)
                all_subject_results[variant['name']]['baseline'].append(result['baseline_acc'])
                all_subject_results[variant['name']]['augmented'].append(result['augmented_acc'])
                all_subject_results[variant['name']]['improvement'].append(result['improvement'])
            except Exception as e:
                print(f"  错误 ({variant['name']}): {e}")
                import traceback
                traceback.print_exc()
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
    output_file = 'outputs/results/paper_experiments/ablation_study_bci2b.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print(f"消融实验结果总结 - BCI2b（跨会话，{n_subjects}个被试平均）")
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
