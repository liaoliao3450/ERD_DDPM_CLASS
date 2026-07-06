#!/usr/bin/env python3
"""
ERD权重和Guidance Scale敏感度分析

分析两个关键参数对跨会话准确率的影响：
1. ERD权重 (erd_weight) - 训练时的ERD损失权重
2. Guidance Scale (guidance_scale) - 推理时的分类器引导强度
"""
import sys, os, torch, numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import accuracy_score

# 统一图像字体配置（与其他图一致，使用默认英文字体）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    pretrain_classifier
)
from data_loader import load_bci2a_data

# ============================================================================
# 配置参数
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SUBJECTS = 9  # 测试所有9个被试
N_CLASSES = 4
N_CHANNELS = 22
N_SAMPLES = 1000
TRIALS_PER_CLASS = 72
FS = 250

# 分类器训练参数
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 1e-3

# 采样参数
DDIM_STEPS = 50
NOISE_SCALE = 1.0
NUM_AUGMENT = 1

# 参数测试范围
ERD_WEIGHTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # ERD权重范围
# Guidance Scale范围：在5.0-7.0之间进行精细搜索，因为5.25时效果最好
GUIDANCE_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 
                   5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0]  # 在5.0-7.0之间精细搜索

# 模型路径配置
MODELS_DIR = 'checkpoints/erd_sensitivity'
DEFAULT_MODEL_PATH = 'checkpoints/trained_ddpm.pt'  # 默认模型路径（如果存在）

# 输出路径
RESULTS_PATH = 'outputs/results/sensitivity_erd_guidance_results.json'
FIGURES_DIR = 'outputs/figures'
RANDOM_SEED = 42

# ============================================================================
# 辅助函数
# ============================================================================

def setup_environment():
    """设置随机种子和环境"""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("="*70)
    print("ERD权重和Guidance Scale敏感度分析")
    print("="*70)
    print(f"设备: {DEVICE}")
    print(f"测试被试数: {N_SUBJECTS}")
    print(f"ERD权重范围: {ERD_WEIGHTS}")
    print(f"Guidance Scale范围: {GUIDANCE_SCALES}")
    print(f"分类器训练轮数: {CLASSIFIER_EPOCHS}")
    print()


def load_ddpm_model(erd_weight=None):
    """
    加载指定ERD权重的DDPM模型
    
    Args:
        erd_weight: ERD权重值，如果为None则加载默认模型
    
    Returns:
        ddpm: 加载的DDPM模型
    """
    if erd_weight is not None:
        model_path = os.path.join(MODELS_DIR, f'ddpm_erd_{erd_weight}.pt')
        if not os.path.exists(model_path):
            print(f"  警告: 找不到ERD权重={erd_weight}的模型: {model_path}")
            print(f"  尝试加载默认模型: {DEFAULT_MODEL_PATH}")
            model_path = DEFAULT_MODEL_PATH
    else:
        model_path = DEFAULT_MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到DDPM模型文件: {model_path}")
    
    print(f"  加载模型: {os.path.basename(model_path)}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    eps_model = MultiScaleCondUNet(channels=N_CHANNELS, num_classes=N_CLASSES).to(DEVICE)
    classifier = EEGClassifier(channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES).to(DEVICE)
    
    target_psd = checkpoint['target_psd'].to(DEVICE)
    target_laterality = checkpoint['target_laterality'].to(DEVICE)
    
    ddpm = ClassDiscriminativeDDPM(
        eps_model, classifier, target_psd, target_laterality,
        n_timesteps=1000, channels=N_CHANNELS, n_samples=N_SAMPLES
    ).to(DEVICE)
    
    # 尝试加载state_dict
    # strict=True: 要求完全匹配，checkpoint和模型的所有参数必须一致
    # strict=False: 允许部分匹配，只加载匹配的参数，忽略缺失或多余的参数（用于模型结构变化时）
    try:
        ddpm.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
            print(f"  警告: 模型checkpoint中缺少某些参数（如spectral_loss_fn的mask），使用非严格模式加载（strict=False）...")
            print(f"  说明: 非严格模式允许只加载匹配的参数，缺失的参数会使用模型初始化时的默认值")
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise
    
    ddpm.eval()
    
    return ddpm


def normalize_generated_data_to_real_stats(real_data, gen_data):
    """
    将生成数据的统计特性对齐到真实数据（全局标准化和重新缩放）
    
    Args:
        real_data: 真实数据 [N, C, T]
        gen_data: 生成数据 [M, C, T]
    
    Returns:
        对齐后的生成数据
    """
    # 计算真实数据的全局统计特性（按通道，axis=(0, 2)）
    real_global_mean = real_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
    real_global_std = real_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
    
    # 计算生成数据的全局统计特性
    gen_global_mean = gen_data.mean(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
    gen_global_std = gen_data.std(axis=(0, 2), keepdims=True)       # (1, n_channels, 1)
    
    # 全局标准化和重新缩放
    gen_data_normalized = (gen_data - gen_global_mean) / (gen_global_std + 1e-8)
    gen_data_matched = gen_data_normalized * real_global_std + real_global_mean
    
    return gen_data_matched


def generate_augmented_data(ddpm, n_per_class, guidance_scale=2.0, real_data=None):
    """
    生成增强数据
    
    Args:
        ddpm: DDPM模型
        n_per_class: 每个类别生成的样本数
        guidance_scale: 分类器引导强度
        real_data: 真实训练数据，用于标准化对齐（可选）
    
    Returns:
        X_gen: 生成的数据
        y_gen: 生成的标签
    """
    ddpm.eval()
    X_gen, y_gen = [], []
    
    with torch.no_grad():
        for c in range(N_CLASSES):
            n_batches = (n_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_per_class - len([y for y in y_gen if y == c]))
                if batch_size <= 0:
                    break
                
                yg = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = ddpm.sample_ddim(batch_size, yg, steps=DDIM_STEPS, guidance_scale=guidance_scale)
                samples = samples * NOISE_SCALE
                
                X_gen.append(samples.cpu().numpy())
                y_gen.extend([c] * batch_size)
    
    X_gen = np.concatenate(X_gen)
    y_gen = np.array(y_gen)
    
    # 如果提供了真实数据，进行标准化对齐
    if real_data is not None:
        X_gen = normalize_generated_data_to_real_stats(real_data, X_gen)
    
    return X_gen, y_gen


def load_baseline_results():
    """
    从evaluate_EEGNET_baseline的结果文件中加载跨会话baseline准确率
    
    Returns:
        包含baseline结果的字典，格式与evaluate_baseline相同
    """
    baseline_file = 'outputs/results/baseline_all_scenarios.json'
    
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(
            f"找不到baseline结果文件: {baseline_file}\n"
            f"请先运行: python experiments/evaluate_EEGNET_baseline.py"
        )
    
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    # 提取跨会话结果
    cross_session = baseline_data['cross_session']
    
    # 转换为与evaluate_baseline相同的格式
    per_subject_accs = {}
    for i, acc in enumerate(cross_session['per_subject']):
        per_subject_accs[f'S{i+1}'] = float(acc)
    
    print("="*70)
    print("加载 Baseline 跨会话结果（来自 evaluate_EEGNET_baseline）")
    print("="*70)
    print(f"平均准确率: {cross_session['mean']*100:.2f}% ± {cross_session['std']*100:.2f}%")
    print(f"每个被试准确率:")
    for subj_key, acc in per_subject_accs.items():
        print(f"  {subj_key}: {acc*100:.2f}%")
    print()
    
    return {
        'per_subject': per_subject_accs,
        'mean': float(cross_session['mean']),
        'std': float(cross_session['std'])
    }


def evaluate_single_config(ddpm, X, y, subjects, sessions, guidance_scale=2.0):
    """
    评估单个参数配置（跨会话准确率）
    
    Args:
        ddpm: DDPM模型
        X: 数据
        y: 标签
        subjects: 被试ID
        sessions: 会话ID
        guidance_scale: 分类器引导强度
    
    Returns:
        包含每个被试准确率和统计信息的字典
    """
    per_subject_accs = {}
    
    for subj_id in range(N_SUBJECTS):
        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"    被试 {subj_id+1}/{N_SUBJECTS}...", end=" ")
        
        # 生成增强数据（传入真实训练数据以进行标准化对齐）
        n_per_class = TRIALS_PER_CLASS * NUM_AUGMENT
        X_gen, y_gen = generate_augmented_data(ddpm, n_per_class, guidance_scale, real_data=X_train)
        
        # 合并训练数据和生成数据
        X_aug = np.concatenate([X_train, X_gen])
        y_aug = np.concatenate([y_train, y_gen])
        
        # 训练分类器
        clf = EEGClassifier(channels=N_CHANNELS, n_samples=N_SAMPLES, num_classes=N_CLASSES).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_aug), torch.LongTensor(y_aug),
            epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
            lr=CLASSIFIER_LR, device=DEVICE, verbose=False
        )
        
        # 评估跨会话准确率
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
        
        acc = accuracy_score(y_test, pred)
        per_subject_accs[f'S{subj_id+1}'] = float(acc)
        print(f"{acc*100:.2f}%")
        
        del clf, X_gen, y_gen, X_aug, y_aug
        torch.cuda.empty_cache()
    
    accs = list(per_subject_accs.values())
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"    平均: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")
    
    return {
        'per_subject': per_subject_accs,
        'mean': float(mean_acc),
        'std': float(std_acc)
    }


def load_existing_results():
    """加载已有的结果文件（如果存在）"""
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  警告: 加载已有结果失败: {e}")
            return None
    return None


def run_sensitivity_analysis(X, y, subjects, sessions):
    """
    运行敏感度分析：分析ERD权重和Guidance Scale的联合影响（二维网格搜索）
    
    注意：如果某个ERD权重的模型不存在，将使用默认模型
    只运行缺失的参数组合，跳过已有结果
    """
    # 从已有结果文件加载Baseline（避免重复实验）
    baseline_results = load_baseline_results()
    baseline_acc = baseline_results['mean']
    
    # 加载已有结果（如果存在）
    existing_results = load_existing_results()
    if existing_results:
        print("="*70)
        print("检测到已有结果文件，将只运行缺失的参数组合")
        print("="*70)
        results = existing_results
        results['baseline'] = baseline_results  # 更新baseline（以防万一）
        existing_keys = set(results.get('grid_search', {}).keys())
        print(f"已有结果数量: {len(existing_keys)}")
    else:
        print("="*70)
        print("未找到已有结果文件，将运行所有参数组合")
        print("="*70)
        results = {}
        results['baseline'] = baseline_results
        results['grid_search'] = {}
        existing_keys = set()
    
    # 二维参数网格搜索：ERD权重 × Guidance Scale
    print(f"\nERD权重范围: {ERD_WEIGHTS}")
    print(f"Guidance Scale范围: {GUIDANCE_SCALES}")
    print(f"总组合数: {len(ERD_WEIGHTS) * len(GUIDANCE_SCALES)}")
    print("注意：如果某个ERD权重的模型不存在，将使用默认模型")
    print()
    
    # 初始化grid_search（如果不存在）
    if 'grid_search' not in results:
     results['grid_search'] = {}
    
    ddpm_cache = {}  # 缓存不同ERD权重的模型
    
    # 统计需要运行的组合
    combinations_to_run = []
    for erd_w in ERD_WEIGHTS:
        for gs in GUIDANCE_SCALES:
            key = f"erd_{erd_w}_gs_{gs}"
            if key not in existing_keys:
                combinations_to_run.append((erd_w, gs, key))
    
    if len(combinations_to_run) == 0:
        print("✓ 所有参数组合已完成，无需重新运行")
        return results
    
    print(f"需要运行的组合数: {len(combinations_to_run)}")
    print(f"跳过的组合数: {len(existing_keys)}\n")
    
    # 运行缺失的组合
    for idx, (erd_w, gs, key) in enumerate(combinations_to_run, 1):
        print(f"[{idx}/{len(combinations_to_run)}] ERD权重={erd_w}, Guidance Scale={gs}")
        print("-" * 70)
        
        # 加载或获取对应ERD权重的模型
        if erd_w not in ddpm_cache:
            try:
                ddpm = load_ddpm_model(erd_weight=erd_w)
                # 检查是否真的加载了对应ERD权重的模型
                model_path = os.path.join(MODELS_DIR, f'ddpm_erd_{erd_w}.pt')
                if not os.path.exists(model_path):
                    # 使用的是默认模型，检查是否已缓存
                    if None not in ddpm_cache:
                        ddpm_cache[None] = ddpm
                    else:
                        ddpm = ddpm_cache[None]
                ddpm_cache[erd_w] = ddpm
            except Exception as e:
                print(f"  警告: 加载ERD权重={erd_w}的模型失败: {e}")
                if None not in ddpm_cache:
                    print(f"  使用默认模型")
                    ddpm_cache[None] = load_ddpm_model(erd_weight=None)
                ddpm_cache[erd_w] = ddpm_cache[None]
        
        ddpm = ddpm_cache[erd_w]
            
        config_results = evaluate_single_config(ddpm, X, y, subjects, sessions, guidance_scale=gs)
            
        # 存储结果
        results['grid_search'][key] = {
                'erd_weight': erd_w,
                'guidance_scale': gs,
                'mean': config_results['mean'],
                'std': config_results['std'],
                'per_subject': config_results['per_subject']
            }
            
        improvement = (config_results['mean'] - baseline_acc) * 100
        print(f"  准确率: {config_results['mean']*100:.2f}% ± {config_results['std']*100:.2f}%")
        print(f"  提升: {improvement:+.2f}%\n")
    
        # 每完成一个组合就保存一次（防止中断丢失数据）
    results['metadata'] = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_subjects_tested': N_SUBJECTS,
        'erd_weights_tested': ERD_WEIGHTS,
        'guidance_scales_tested': GUIDANCE_SCALES,
            'total_combinations': len(ERD_WEIGHTS) * len(GUIDANCE_SCALES),
        'note': 'ERD权重和Guidance Scale二维网格搜索，生成数据已进行全局标准化对齐'
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 完成！已更新结果文件: {RESULTS_PATH}")
    
    return results


def plot_sensitivity_heatmap(results, baseline_acc):
    """
    绘制ERD权重和Guidance Scale的二维敏感度热力图
    
    Args:
        results: 结果字典，包含grid_search
        baseline_acc: baseline准确率
    """
    if 'grid_search' not in results:
        print("  警告: 没有找到grid_search结果，跳过热力图绘制")
        return
    
    # 提取数据
    erd_weights = sorted(ERD_WEIGHTS)
    guidance_scales = sorted(GUIDANCE_SCALES)
    
    # 构建准确率矩阵
    acc_matrix = np.zeros((len(erd_weights), len(guidance_scales)))
    
    for i, erd_w in enumerate(erd_weights):
        for j, gs in enumerate(guidance_scales):
            key = f"erd_{erd_w}_gs_{gs}"
            if key in results['grid_search']:
                acc_matrix[i, j] = results['grid_search'][key]['mean'] * 100
            else:
                acc_matrix[i, j] = np.nan
    
    baseline_acc_pct = baseline_acc * 100
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=baseline_acc_pct - 5, vmax=acc_matrix.max() + 2)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(guidance_scales)))
    ax.set_yticks(np.arange(len(erd_weights)))
    ax.set_xticklabels([f'{gs:.1f}' for gs in guidance_scales])
    ax.set_yticklabels([f'{ew:.1f}' for ew in erd_weights])
    
    # 添加数值标注
    for i in range(len(erd_weights)):
        for j in range(len(guidance_scales)):
            if not np.isnan(acc_matrix[i, j]):
                text = ax.text(j, i, f'{acc_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=12)
    
    # 标签和标题
    ax.set_xlabel('Guidance Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('ERD Weight', fontsize=12, fontweight='bold')
    ax.set_title('Joint Sensitivity Analysis: ERD Weight × Guidance Scale\nCross-Session Accuracy (%)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cross-Session Accuracy (%)', fontsize=12, fontweight='bold')
    
    # 添加baseline参考线
    ax.axhline(-0.5, color='#A23B72', linestyle='--', linewidth=2, 
               label=f'Baseline: {baseline_acc_pct:.2f}%', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, 'sensitivity_heatmap_erd_guidance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    
    # 同时保存PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"  保存PDF: {pdf_path}")
    
    plt.close()


def plot_all_sensitivity_curves(results):
    """绘制敏感度分析图（二维热力图）"""
    print("\n" + "="*70)
    print("生成敏感度分析图")
    print("="*70)
    
    baseline_acc = results['baseline']['mean']
    
    # 绘制二维热力图
    print(f"\n绘制 ERD权重 × Guidance Scale 二维热力图...")
    plot_sensitivity_heatmap(results, baseline_acc)
    
    print("\n✓ 敏感度分析图生成完成\n")


def print_summary(results):
    """打印总结"""
    print("\n" + "="*70)
    print("敏感度分析总结")
    print("="*70)
    
    baseline_acc = results['baseline']['mean']
    print(f"\nBaseline (无数据增强): {baseline_acc*100:.2f}%")
    
    # 从网格搜索中找到最优组合
    if 'grid_search' in results:
        print(f"\n二维参数网格搜索结果:")
        best_erd = None
        best_gs = None
        best_acc = 0
        
        for key, value in results['grid_search'].items():
            mean_acc = value['mean']
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_erd = value['erd_weight']
                best_gs = value['guidance_scale']
        
        improvement = (best_acc - baseline_acc) * 100
        print(f"  最优组合: ERD权重={best_erd}, Guidance Scale={best_gs}")
        print(f"  准确率: {best_acc*100:.2f}% (提升: {improvement:+.2f}%)")
        
        # 统计信息
        all_accs = [v['mean'] for v in results['grid_search'].values()]
        print(f"\n  所有组合统计:")
        print(f"    平均准确率: {np.mean(all_accs)*100:.2f}%")
        print(f"    最高准确率: {np.max(all_accs)*100:.2f}%")
        print(f"    最低准确率: {np.min(all_accs)*100:.2f}%")
        print(f"    超过baseline的组合数: {sum(1 for acc in all_accs if acc > baseline_acc)}/{len(all_accs)}")
    
    print("\n" + "="*70)
    print(f"结果: {RESULTS_PATH}")
    print(f"图表: {FIGURES_DIR}/sensitivity_heatmap_erd_guidance.png")
    print("="*70 + "\n")


def main():
    """主函数"""
    try:
        setup_environment()
        
        print("加载数据...")
        X, y, subjects, sessions = load_bci2a_data()
        print()
        
        # 运行敏感度分析（内部会加载模型）
        results = run_sensitivity_analysis(X, y, subjects, sessions)
        
        plot_all_sensitivity_curves(results)
        
        print_summary(results)
        
        print("✅ ERD权重和Guidance Scale敏感度分析完成！\n")
        print("注意: 生成数据已进行全局标准化对齐，以匹配真实数据分布。\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
