"""
正确的GAN/VAE训练和评估流程
对比：Baseline vs 增强后的分类准确率
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# 添加模型路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from data_loader import load_bci2a_data

# 导入GAN模型
gan_path = str(PROJECT_ROOT / 'core' / 'models' / 'gan')
if gan_path not in sys.path:
    sys.path.insert(0, gan_path)
from model import Gen1D, Disc1D

# 导入VAE模型
vae_path = str(PROJECT_ROOT / 'core' / 'models' / 'vae')
if vae_path not in sys.path:
    sys.path.insert(0, vae_path)
from vae_model import VAE1D

# 导入分类器
ddpm_path = str(PROJECT_ROOT / 'core' / 'models' / 'ddpm')
if ddpm_path not in sys.path:
    sys.path.insert(0, ddpm_path)
from class_discriminative import EEGClassifier, pretrain_classifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_gan(X, y, epochs=500, batch_size=64):
    """训练GAN"""
    print("\n训练GAN...")
    print(f"  数据: {X.shape}, Epochs: {epochs}")
    
    G = Gen1D(z_dim=128, out_channels=22, out_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
    D = Disc1D(in_channels=22, in_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
    
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    n_samples = len(X)
    
    for epoch in range(epochs):
        G.train()
        D.train()
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            real_X = X_tensor[batch_indices].to(DEVICE)
            real_y = y_tensor[batch_indices].to(DEVICE)
            bs = len(real_X)
            
            # 训练判别器
            opt_D.zero_grad()
            real_pred = D(real_X, real_y)
            loss_D_real = -torch.mean(real_pred)
            
            z = torch.randn(bs, 128, device=DEVICE)
            fake_X = G(z, real_y)
            fake_pred = D(fake_X.detach(), real_y)
            loss_D_fake = torch.mean(fake_pred)
            
            alpha = torch.rand(bs, 1, 1, device=DEVICE)
            interpolates = (alpha * real_X + (1 - alpha) * fake_X.detach()).requires_grad_(True)
            d_interpolates = D(interpolates, real_y)
            gradients = torch.autograd.grad(
                outputs=d_interpolates, inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True, retain_graph=True
            )[0]
            gradient_penalty = ((gradients.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean() * 10
            
            loss_D = loss_D_real + loss_D_fake + gradient_penalty
            loss_D.backward()
            opt_D.step()
            
            # 训练生成器
            if i % (5 * batch_size) == 0:
                opt_G.zero_grad()
                z = torch.randn(bs, 128, device=DEVICE)
                fake_X = G(z, real_y)
                fake_pred = D(fake_X, real_y)
                loss_G = -torch.mean(fake_pred)
                loss_G.backward()
                opt_G.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    
    print("  ✓ GAN训练完成")
    os.makedirs('checkpoints/gan', exist_ok=True)
    torch.save({'G': G.state_dict(), 'D': D.state_dict()}, 'checkpoints/gan/gan_retrained.pt')
    return G

def train_vae(X, y, epochs=500, batch_size=64):
    """训练VAE"""
    print("\n训练VAE...")
    print(f"  数据: {X.shape}, Epochs: {epochs}")
    
    vae = VAE1D(channels=22, length=1000, latent_dim=128, cond_dim=32, num_classes=4).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    n_samples = len(X)
    
    for epoch in range(epochs):
        vae.train()
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_tensor[batch_indices].to(DEVICE)
            batch_y = y_tensor[batch_indices].to(DEVICE)
            
            optimizer.zero_grad()
            recon_X, mu, logvar = vae(batch_X, batch_y)
            recon_loss = nn.functional.mse_loss(recon_X, batch_X, reduction='sum') / len(batch_X)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch_X)
            loss = recon_loss + kl_loss * 0.1
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    
    print("  ✓ VAE训练完成")
    os.makedirs('checkpoints/vae', exist_ok=True)
    torch.save({'model': vae.state_dict()}, 'checkpoints/vae/vae_retrained.pt')
    return vae

def generate_samples(model, model_type, n_samples=2000):
    """生成样本"""
    print(f"  生成{n_samples}个样本...")
    model.eval()
    gen_X, gen_y = [], []
    
    with torch.no_grad():
        for c in range(4):
            n_per_class = n_samples // 4
            z = torch.randn(n_per_class, 128, device=DEVICE)
            y_batch = torch.full((n_per_class,), c, dtype=torch.long, device=DEVICE)
            
            if model_type == 'GAN':
                samples = model(z, y_batch)
            else:  # VAE
                samples = model.decode(z, y_batch)
            
            gen_X.append(samples.cpu().numpy())
            gen_y.extend([c] * n_per_class)
    
    gen_X = np.concatenate(gen_X, axis=0)
    gen_y = np.array(gen_y)
    print(f"  ✓ 生成完成: {gen_X.shape}")
    return gen_X, gen_y

def evaluate_augmentation(X, y, subjects, sessions, X_gen, y_gen, method_name):
    """
    评估数据增强效果
    对比：Baseline (只用原始数据) vs Augmented (原始+生成数据)
    使用EEGClassifier进行评估
    """
    print(f"\n{'='*60}")
    print(f"评估 {method_name} 数据增强效果")
    print(f"{'='*60}")
    
    results = {}
    scenarios = [
        ('within_subject', 'Within-Subject'),
        ('cross_session', 'Cross-Session'),
        ('cross_subject', 'Cross-Subject')
    ]
    
    for scenario_key, scenario_name in scenarios:
        print(f"\n{scenario_name}:")
        
        baseline_accs = []
        augmented_accs = []
        n_subjects = len(np.unique(subjects))
        
        for subj_id in range(n_subjects):
            # 准备训练/测试数据
            if scenario_key == 'within_subject':
                mask = subjects == subj_id
                X_subj = X[mask]
                y_subj = y[mask]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subj, y_subj, test_size=0.2, random_state=42, stratify=y_subj
                )
            elif scenario_key == 'cross_session':
                train_mask = (subjects == subj_id) & (sessions == 0)
                test_mask = (subjects == subj_id) & (sessions == 1)
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
            else:  # cross_subject
                train_mask = (subjects != subj_id) & (sessions == 0)
                test_mask = (subjects == subj_id) & (sessions == 0)
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
            
            # Baseline: 只用原始数据
            clf_baseline = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
            clf_baseline = pretrain_classifier(
                clf_baseline, 
                torch.FloatTensor(X_train), 
                torch.LongTensor(y_train),
                epochs=100, batch_size=32, lr=1e-3, device=DEVICE, verbose=False
            )
            clf_baseline.eval()
            with torch.no_grad():
                pred_baseline = clf_baseline(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
            acc_baseline = accuracy_score(y_test, pred_baseline)
            baseline_accs.append(acc_baseline)
            
            # Augmented: 原始数据 + 生成数据
            n_gen = len(X_train)  # 生成与训练集相同数量的样本
            indices = np.random.choice(len(X_gen), n_gen, replace=True)
            X_gen_subset = X_gen[indices]
            y_gen_subset = y_gen[indices]
            
            X_combined = np.concatenate([X_train, X_gen_subset], axis=0)
            y_combined = np.concatenate([y_train, y_gen_subset], axis=0)
            
            clf_augmented = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
            clf_augmented = pretrain_classifier(
                clf_augmented,
                torch.FloatTensor(X_combined),
                torch.LongTensor(y_combined),
                epochs=100, batch_size=32, lr=1e-3, device=DEVICE, verbose=False
            )
            clf_augmented.eval()
            with torch.no_grad():
                pred_augmented = clf_augmented(torch.FloatTensor(X_test).to(DEVICE)).argmax(1).cpu().numpy()
            acc_augmented = accuracy_score(y_test, pred_augmented)
            augmented_accs.append(acc_augmented)
            
            improvement = acc_augmented - acc_baseline
            marker = "✓" if improvement > 0 else "✗"
            print(f"  被试{subj_id+1}: Baseline={acc_baseline:.4f}, Augmented={acc_augmented:.4f}, Δ={improvement:+.4f} {marker}")
        
        baseline_mean = np.mean(baseline_accs)
        baseline_std = np.std(baseline_accs)
        augmented_mean = np.mean(augmented_accs)
        augmented_std = np.std(augmented_accs)
        improvement = augmented_mean - baseline_mean
        
        print(f"\n  Baseline:  {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Augmented: {augmented_mean:.4f} ± {augmented_std:.4f}")
        print(f"  提升:      {improvement:+.4f}")
        
        results[scenario_key] = {
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'augmented_mean': float(augmented_mean),
            'augmented_std': float(augmented_std),
            'improvement': float(improvement),
            'baseline_per_subject': [float(x) for x in baseline_accs],
            'augmented_per_subject': [float(x) for x in augmented_accs]
        }
    
    return results

def main():
    print("="*60)
    print("GAN/VAE 数据增强实验")
    print("="*60)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    print("加载BCI数据...")
    X, y, subjects, sessions = load_bci2a_data()
    print(f"  数据: {X.shape}, 被试数: {len(np.unique(subjects))}")
    
    # 使用Session 0训练生成模型
    train_mask = sessions == 0
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    print(f"  生成模型训练数据: {X_train_all.shape}\n")
    
    # ==================== GAN ====================
    print("="*60)
    print("1. GAN 实验")
    print("="*60)
    
    gan_model = train_gan(X_train_all, y_train_all, epochs=500, batch_size=64)
    gan_X, gan_y = generate_samples(gan_model, 'GAN', n_samples=2000)
    
    # 保存GAN样本
    os.makedirs('outputs/gan_samples', exist_ok=True)
    np.save('outputs/gan_samples/gan_samples.npy', gan_X)
    np.save('outputs/gan_samples/gan_labels.npy', gan_y)
    
    # 评估GAN增强效果
    gan_results = evaluate_augmentation(X, y, subjects, sessions, gan_X, gan_y, 'GAN')
    
    # 保存GAN结果
    gan_output = {
        'method': 'GAN',
        'within_subject': gan_results['within_subject'],
        'cross_session': gan_results['cross_session'],
        'cross_subject': gan_results['cross_subject']
    }
    os.makedirs('outputs/results', exist_ok=True)
    with open('outputs/results/gan_all_scenarios.json', 'w') as f:
        json.dump(gan_output, f, indent=2)
    print("\n✓ GAN结果已保存")
    
    # ==================== VAE ====================
    print("\n" + "="*60)
    print("2. VAE 实验")
    print("="*60)
    
    vae_model = train_vae(X_train_all, y_train_all, epochs=500, batch_size=64)
    vae_X, vae_y = generate_samples(vae_model, 'VAE', n_samples=2000)
    
    # 保存VAE样本
    os.makedirs('outputs/vae_samples', exist_ok=True)
    np.save('outputs/vae_samples/vae_samples.npy', vae_X)
    np.save('outputs/vae_samples/vae_labels.npy', vae_y)
    
    # 评估VAE增强效果
    vae_results = evaluate_augmentation(X, y, subjects, sessions, vae_X, vae_y, 'VAE')
    
    # 保存VAE结果
    vae_output = {
        'method': 'VAE',
        'within_subject': vae_results['within_subject'],
        'cross_session': vae_results['cross_session'],
        'cross_subject': vae_results['cross_subject']
    }
    with open('outputs/results/vae_all_scenarios.json', 'w') as f:
        json.dump(vae_output, f, indent=2)
    print("\n✓ VAE结果已保存")
    
    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    
    print("\nGAN 数据增强效果:")
    for scenario in ['within_subject', 'cross_session', 'cross_subject']:
        r = gan_results[scenario]
        print(f"  {scenario}:")
        print(f"    Baseline:  {r['baseline_mean']:.4f} ± {r['baseline_std']:.4f}")
        print(f"    Augmented: {r['augmented_mean']:.4f} ± {r['augmented_std']:.4f}")
        print(f"    提升:      {r['improvement']:+.4f}")
    
    print("\nVAE 数据增强效果:")
    for scenario in ['within_subject', 'cross_session', 'cross_subject']:
        r = vae_results[scenario]
        print(f"  {scenario}:")
        print(f"    Baseline:  {r['baseline_mean']:.4f} ± {r['baseline_std']:.4f}")
        print(f"    Augmented: {r['augmented_mean']:.4f} ± {r['augmented_std']:.4f}")
        print(f"    提升:      {r['improvement']:+.4f}")
    
    print("\n下一步:")
    print("  python experiments/paper_experiments/3_compare_all_augmentation_methods.py")

if __name__ == '__main__':
    main()
