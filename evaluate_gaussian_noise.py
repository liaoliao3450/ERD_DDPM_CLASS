#!/usr/bin/env python3
"""
Gaussian Noise方法评估

包含三种测试场景：
1. Within-Subject (被试内)
2. Cross-Session (跨会话)
3. Cross-Subject (跨被试, LOSO)

通过添加高斯噪声进行数据增强
"""
import sys, os, torch, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')
from class_discriminative import EEGClassifier, pretrain_classifier
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_noise_augmentation(X_train, y_train, n_samples_per_class, noise_level=0.1):
    """
    高斯噪声数据增强
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        n_samples_per_class: 每类生成的样本数
        noise_level: 噪声水平（相对于数据标准差）
    """
    gen_X, gen_y = [], []
    
    for c in range(4):
        class_data = X_train[y_train == c]
        # 计算该类数据的标准差
        class_std = class_data.std(axis=0)
        
        for _ in range(n_samples_per_class):
            # 随机选择一个基础样本
            base = class_data[np.random.randint(len(class_data))]
            
            # 添加高斯噪声
            noise = np.random.randn(22, 1000) * class_std * noise_level
            sample = base + noise
            
            gen_X.append(sample)
            gen_y.append(c)
    
    return np.array(gen_X), np.array(gen_y)

def within_subject_test(X, y, subjects):
    """被试内测试"""
    print("\n" + "="*70)
    print("1. Within-Subject 测试")
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
        
        # 高斯噪声生成
        gen_X, gen_y = gaussian_noise_augmentation(X_train, y_train, samples_per_class)
        
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

def cross_session_test(X, y, subjects, sessions):
    """跨会话测试"""
    print("\n" + "="*70)
    print("2. Cross-Session 测试")
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
        
        # 高斯噪声生成
        gen_X, gen_y = gaussian_noise_augmentation(X_train, y_train, samples_per_class)
        
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

def cross_subject_test(X, y, subjects, sessions):
    """跨被试测试（LOSO）"""
    print("\n" + "="*70)
    print("3. Cross-Subject 测试 (LOSO)")
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
        
        # 高斯噪声生成
        gen_X, gen_y = gaussian_noise_augmentation(X_train, y_train, samples_per_class)
        
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

def main():
    print("="*70)
    print("Gaussian Noise方法评估")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    X, y, subjects, sessions = load_bci2a_data()
    print(f"被试数: {len(np.unique(subjects))}")
    print(f"会话数: {len(np.unique(sessions))}")
    
    # 三种测试
    results = {}
    results['within_subject'] = within_subject_test(X, y, subjects)
    results['cross_session'] = cross_session_test(X, y, subjects, sessions)
    results['cross_subject'] = cross_subject_test(X, y, subjects, sessions)
    
    # 汇总
    print("\n" + "="*70)
    print("Gaussian Noise 最终结果汇总")
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
        'method': 'Gaussian Noise',
        'within_subject': results['within_subject'],
        'cross_session': results['cross_session'],
        'cross_subject': results['cross_subject']
    }
    
    with open('outputs/results/gaussian_noise_all_scenarios.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n结果已保存到: outputs/results/gaussian_noise_all_scenarios.json")
    print("\n" + "="*70)
    print("Gaussian Noise评估完成！")
    print("="*70)

if __name__ == '__main__':
    main()
