#!/usr/bin/env python3
"""
GAN和VAE方法评估

包含三种测试场景：
1. Within-Subject (被试内)
2. Cross-Session (跨会话)
3. Cross-Subject (跨被试, LOSO)
"""
import sys, os, torch, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入模型
import importlib.util

# 加载GAN模型
gan_model_path = PROJECT_ROOT / 'core' / 'models' / 'gan' / 'model.py'
spec = importlib.util.spec_from_file_location("gan_model", gan_model_path)
gan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan_module)
Gen1D = gan_module.Gen1D

# 加载VAE模型（使用 VAE1D）
vae_model_path = PROJECT_ROOT / 'core' / 'models' / 'vae' / 'vae_model.py'
spec = importlib.util.spec_from_file_location("vae_model", vae_model_path)
vae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vae_module)

# 加载DDPM工具
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'ddpm'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))
from class_discriminative import EEGClassifier, pretrain_classifier
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_gan_model():
    """加载训练好的GAN模型"""
    # 使用与正确训练脚本一致的重训练权重
    ckpt_path = 'checkpoints/gan/gan_retrained.pt'
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 未找到GAN checkpoint: {ckpt_path}")
        return None
    
    try:
        # 结构需与训练时保持一致
        G = Gen1D(z_dim=128, out_channels=22, out_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(state['G'])
        G.eval()
        print(f"  ✅ GAN加载成功: {ckpt_path}")
        return G
    except Exception as e:
        print(f"  ❌ GAN加载失败: {e}")
        return None

def load_vae_model():
    """加载训练好的VAE模型"""
    ckpt_path = 'checkpoints/vae/vae_retrained.pt'
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 未找到VAE checkpoint: {ckpt_path}")
        return None
    
    try:
        # 与论文实验中使用的配置保持一致
        vae = vae_module.VAE1D(channels=22, length=1000, latent_dim=128, cond_dim=32, num_classes=4).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        vae.load_state_dict(state['model'])
        vae.eval()
        print(f"  ✅ VAE加载成功")
        return vae
    except Exception as e:
        print(f"  ❌ VAE加载失败: {e}")
        return None

def generate_gan_samples(G, n_samples_per_class):
    """使用GAN生成样本"""
    gen_X, gen_y = [], []
    
    with torch.no_grad():
        for c in range(4):
            n_batches = (n_samples_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_samples_per_class - len([y for y in gen_y if y == c]))
                if batch_size <= 0:
                    break
                z = torch.randn(batch_size, 128, device=DEVICE)
                yg = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = G(z, yg)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)
    
    return np.concatenate(gen_X), np.array(gen_y)

def generate_vae_samples(vae, n_samples_per_class):
    """使用VAE生成样本"""
    gen_X, gen_y = [], []
    
    with torch.no_grad():
        for c in range(4):
            n_batches = (n_samples_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_samples_per_class - len([y for y in gen_y if y == c]))
                if batch_size <= 0:
                    break
                z = torch.randn(batch_size, 128, device=DEVICE)
                yg = torch.full((batch_size,), c, dtype=torch.long, device=DEVICE)
                samples = vae.decode(z, yg)
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * batch_size)
    
    return np.concatenate(gen_X), np.array(gen_y)

def within_subject_test(model, model_name, generate_func, X, y, subjects):
    """被试内测试"""
    print(f"\n{'='*70}")
    print(f"1. Within-Subject 测试 - {model_name}")
    print("="*70)
    
    results = []
    n_subjects = len(np.unique(subjects))
    
    for subj_id in range(n_subjects):
        print(f"\n被试 {subj_id+1}/{n_subjects}:")
        
        mask = subjects == subj_id
        X_subj = X[mask]
        y_subj = y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
        )
        
        samples_per_class = len(X_train) // 4
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}, 生成: {samples_per_class}×4")
        
        # 生成数据
        gen_X, gen_y = generate_func(model, samples_per_class)
        
        # 合并
        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])
        
        # 训练
        clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=100, batch_size=32, lr=1e-3, device=DEVICE, verbose=False
        )
        
        # 评估
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
        
        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc*100:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    print(f"\n平均: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(acc) for acc in results]
    }

def cross_session_test(model, model_name, generate_func, X, y, subjects, sessions):
    """跨会话测试"""
    print(f"\n{'='*70}")
    print(f"2. Cross-Session 测试 - {model_name}")
    print("="*70)
    
    results = []
    n_subjects = len(np.unique(subjects))
    
    for subj_id in range(n_subjects):
        print(f"\n被试 {subj_id+1}/{n_subjects}:")
        
        train_mask = (subjects == subj_id) & (sessions == 0)
        test_mask = (subjects == subj_id) & (sessions == 1)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        samples_per_class = len(X_train) // 4
        print(f"  Session 1训练: {len(X_train)}, Session 2测试: {len(X_test)}, 生成: {samples_per_class}×4")
        
        # 生成数据
        gen_X, gen_y = generate_func(model, samples_per_class)
        
        # 合并
        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])
        
        # 训练
        clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=100, batch_size=32, lr=1e-3, device=DEVICE, verbose=False
        )
        
        # 评估
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
        
        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc*100:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    print(f"\n平均: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(acc) for acc in results]
    }

def cross_subject_test(model, model_name, generate_func, X, y, subjects, sessions):
    """跨被试测试（LOSO）"""
    print(f"\n{'='*70}")
    print(f"3. Cross-Subject 测试 (LOSO) - {model_name}")
    print("="*70)
    
    results = []
    n_subjects = len(np.unique(subjects))
    
    for test_subj in range(n_subjects):
        print(f"\n测试被试 {test_subj+1}/{n_subjects}:")
        
        train_mask = (subjects != test_subj) & (sessions == 0)
        test_mask = (subjects == test_subj) & (sessions == 0)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        samples_per_class = len(X_train) // 4
        print(f"  训练: {len(X_train)} (8个被试), 测试: {len(X_test)}, 生成: {samples_per_class}×4")
        
        # 生成数据
        gen_X, gen_y = generate_func(model, samples_per_class)
        
        # 合并
        X_train_aug = np.concatenate([X_train, gen_X])
        y_train_aug = np.concatenate([y_train, gen_y])
        
        # 训练
        clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
        clf = pretrain_classifier(
            clf, torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug),
            epochs=100, batch_size=32, lr=1e-3, device=DEVICE, verbose=False
        )
        
        # 评估
        clf.eval()
        with torch.no_grad():
            pred = clf(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
        
        acc = accuracy_score(y_test, pred)
        results.append(acc)
        print(f"  准确率: {acc*100:.2f}%")
    
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    print(f"\n平均: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    return {
        'mean': float(mean_acc),
        'std': float(std_acc),
        'per_subject': [float(acc) for acc in results]
    }

def evaluate_model(model_name, model, generate_func):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"📊 {model_name}方法评估")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    if model is None:
        print(f"❌ {model_name}模型加载失败，跳过评估")
        return None
    
    # 加载数据
    X, y, subjects, sessions = load_bci2a_data()
    print(f"被试数: {len(np.unique(subjects))}")
    print(f"会话数: {len(np.unique(sessions))}")
    
    # 三种测试
    results = {}
    results['within_subject'] = within_subject_test(model, model_name, generate_func, X, y, subjects)
    results['cross_session'] = cross_session_test(model, model_name, generate_func, X, y, subjects, sessions)
    results['cross_subject'] = cross_subject_test(model, model_name, generate_func, X, y, subjects, sessions)
    
    # 汇总
    print(f"\n{'='*70}")
    print(f"📊 {model_name} 最终结果汇总")
    print("="*70)
    
    print(f"\n{'场景':<20} {'平均准确率':<15} {'标准差':<10}")
    print("-" * 50)
    print(f"{'Within-Subject':<20} {results['within_subject']['mean']*100:>6.2f}%        {results['within_subject']['std']*100:>6.2f}%")
    print(f"{'Cross-Session':<20} {results['cross_session']['mean']*100:>6.2f}%        {results['cross_session']['std']*100:>6.2f}%")
    print(f"{'Cross-Subject':<20} {results['cross_subject']['mean']*100:>6.2f}%        {results['cross_subject']['std']*100:>6.2f}%")
    
    # 保存
    import json
    os.makedirs('outputs/results', exist_ok=True)
    
    final_results = {
        'method': model_name,
        'within_subject': results['within_subject'],
        'cross_session': results['cross_session'],
        'cross_subject': results['cross_subject']
    }
    
    filename = f"{model_name.lower().replace(' ', '_')}_all_scenarios.json"
    with open(f'outputs/results/{filename}', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 结果已保存到: outputs/results/{filename}")
    print(f"\n{'='*70}")
    print(f"✅ {model_name}评估完成！")
    print("="*70)
    
    return results

def main():
    print("="*70)
    print("📊 GAN和VAE方法评估")
    print("="*70)
    
    # 评估GAN
    print("\n[1/2] 加载GAN模型...")
    gan_model = load_gan_model()
    if gan_model:
        evaluate_model("GAN", gan_model, generate_gan_samples)
    
    # 评估VAE
    print("\n[2/2] 加载VAE模型...")
    vae_model = load_vae_model()
    if vae_model:
        evaluate_model("VAE", vae_model, generate_vae_samples)
    
    print("\n" + "="*70)
    print("✅ 所有评估完成！")
    print("="*70)

if __name__ == '__main__':
    main()
