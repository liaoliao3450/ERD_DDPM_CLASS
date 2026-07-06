"""
重新训练GAN和VAE，生成样本并评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# 添加模型路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'gan'))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'vae'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from model import Gen1D, Disc1D
from vae_model import VAE1D
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_gan(X, y, epochs=50, batch_size=32):
    """训练GAN"""
    print("\n" + "="*60)
    print("训练GAN")
    print("="*60)
    
    # 创建模型
    G = Gen1D(z_dim=128, out_channels=22, out_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
    D = Disc1D(in_channels=22, in_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
    
    # 优化器
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # 转换数据
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    n_samples = len(X)
    
    print(f"训练数据: {X.shape}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"设备: {DEVICE}\n")
    
    for epoch in range(epochs):
        G.train()
        D.train()
        
        # 随机打乱
        indices = torch.randperm(n_samples)
        
        epoch_loss_D = 0
        epoch_loss_G = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            real_X = X_tensor[batch_indices].to(DEVICE)
            real_y = y_tensor[batch_indices].to(DEVICE)
            current_batch_size = len(real_X)
            
            # ==================== 训练判别器 ====================
            opt_D.zero_grad()
            
            # 真实样本
            real_pred = D(real_X, real_y)
            loss_D_real = -torch.mean(real_pred)
            
            # 生成样本
            z = torch.randn(current_batch_size, 128, device=DEVICE)
            fake_X = G(z, real_y)
            fake_pred = D(fake_X.detach(), real_y)
            loss_D_fake = torch.mean(fake_pred)
            
            # 梯度惩罚
            alpha = torch.rand(current_batch_size, 1, 1, device=DEVICE)
            interpolates = (alpha * real_X + (1 - alpha) * fake_X.detach()).requires_grad_(True)
            d_interpolates = D(interpolates, real_y)
            
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True
            )[0]
            
            gradients = gradients.view(current_batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            
            loss_D = loss_D_real + loss_D_fake + gradient_penalty
            loss_D.backward()
            opt_D.step()
            
            epoch_loss_D += loss_D.item()
            
            # ==================== 训练生成器 ====================
            if n_batches % 5 == 0:  # 每5个batch训练一次生成器
                opt_G.zero_grad()
                
                z = torch.randn(current_batch_size, 128, device=DEVICE)
                fake_X = G(z, real_y)
                fake_pred = D(fake_X, real_y)
                loss_G = -torch.mean(fake_pred)
                
                loss_G.backward()
                opt_G.step()
                
                epoch_loss_G += loss_G.item()
            
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - D_loss: {epoch_loss_D/n_batches:.4f}, G_loss: {epoch_loss_G/(n_batches//5):.4f}")
    
    print("\n✅ GAN训练完成")
    
    # 保存模型
    os.makedirs('checkpoints/gan', exist_ok=True)
    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict()
    }, 'checkpoints/gan/gan_retrained.pt')
    print("模型已保存到: checkpoints/gan/gan_retrained.pt")
    
    return G

def train_vae(X, y, epochs=50, batch_size=32):
    """训练VAE"""
    print("\n" + "="*60)
    print("训练VAE")
    print("="*60)
    
    # 创建模型
    vae = VAE1D(channels=22, length=1000, latent_dim=128, cond_dim=32, num_classes=4).to(DEVICE)
    
    # 优化器
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # 转换数据
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    n_samples = len(X)
    
    print(f"训练数据: {X.shape}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"设备: {DEVICE}\n")
    
    for epoch in range(epochs):
        vae.train()
        
        # 随机打乱
        indices = torch.randperm(n_samples)
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_tensor[batch_indices].to(DEVICE)
            batch_y = y_tensor[batch_indices].to(DEVICE)
            
            optimizer.zero_grad()
            
            # 前向传播
            recon_X, mu, logvar = vae(batch_X, batch_y)
            
            # 重构损失
            recon_loss = nn.functional.mse_loss(recon_X, batch_X, reduction='sum') / len(batch_X)
            
            # KL散度
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch_X)
            
            # 总损失
            loss = recon_loss + kl_loss * 0.1
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/n_batches:.4f}")
    
    print("\n✅ VAE训练完成")
    
    # 保存模型
    os.makedirs('checkpoints/vae', exist_ok=True)
    torch.save({
        'model': vae.state_dict()
    }, 'checkpoints/vae/vae_retrained.pt')
    print("模型已保存到: checkpoints/vae/vae_retrained.pt")
    
    return vae

def generate_samples(model, model_type, n_samples=2000):
    """生成样本"""
    print(f"\n生成{model_type}样本 (n={n_samples})...")
    
    model.eval()
    gen_X = []
    gen_y = []
    
    batch_size = 50
    
    with torch.no_grad():
        for c in range(4):
            n_per_class = n_samples // 4
            n_batches = (n_per_class + batch_size - 1) // batch_size
            
            for i in range(n_batches):
                current_size = min(batch_size, n_per_class - len([y for y in gen_y if y == c]))
                if current_size <= 0:
                    break
                
                z = torch.randn(current_size, 128, device=DEVICE)
                y_batch = torch.full((current_size,), c, dtype=torch.long, device=DEVICE)
                
                if model_type == 'GAN':
                    samples = model(z, y_batch)
                else:  # VAE
                    samples = model.decode(z, y_batch)
                
                gen_X.append(samples.cpu().numpy())
                gen_y.extend([c] * current_size)
    
    gen_X = np.concatenate(gen_X, axis=0)
    gen_y = np.array(gen_y)
    
    print(f"  生成完成: {gen_X.shape}")
    print(f"  数据范围: [{gen_X.min():.3f}, {gen_X.max():.3f}]")
    print(f"  均值: {gen_X.mean():.3f}, 标准差: {gen_X.std():.3f}")
    
    return gen_X, gen_y

def evaluate_method(X, y, subjects, sessions, X_gen, y_gen, method_name):
    """评估增强方法"""
    print(f"\n" + "="*60)
    print(f"评估 {method_name}")
    print("="*60)
    
    results = {}
    
    # 三个场景
    scenarios = [
        ('within_subject', 'Within-Subject'),
        ('cross_session', 'Cross-Session'),
        ('cross_subject', 'Cross-Subject')
    ]
    
    for scenario_key, scenario_name in scenarios:
        print(f"\n{scenario_name}:")
        
        per_subject = []
        n_subjects = len(np.unique(subjects))
        
        for subj_id in range(n_subjects):
            if scenario_key == 'within_subject':
                mask = subjects == subj_id
                X_subj = X[mask]
                y_subj = y[mask]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subj, y_subj, test_size=0.3, random_state=42, stratify=y_subj
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
            
            # 选择生成样本
            n_gen = len(X_train)
            indices = np.random.choice(len(X_gen), n_gen, replace=True)
            X_gen_subset = X_gen[indices]
            y_gen_subset = y_gen[indices]
            
            # 合并数据
            X_combined = np.concatenate([X_train, X_gen_subset], axis=0)
            y_combined = np.concatenate([y_train, y_gen_subset], axis=0)
            
            # 训练分类器
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_combined.reshape(len(X_combined), -1), y_combined)
            
            # 评估
            acc = clf.score(X_test.reshape(len(X_test), -1), y_test)
            per_subject.append(acc)
            
            print(f"  被试 {subj_id+1}: {acc:.4f}")
        
        mean_acc = np.mean(per_subject)
        std_acc = np.std(per_subject)
        
        print(f"  平均: {mean_acc:.4f} ± {std_acc:.4f}")
        
        results[scenario_key] = {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'per_subject': [float(x) for x in per_subject]
        }
    
    return results

def main():
    print("="*60)
    print("重新训练和评估GAN/VAE")
    print("="*60)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    print("加载BCI数据...")
    X, y, subjects, sessions = load_bci2a_data()
    print(f"数据形状: {X.shape}")
    print(f"被试数: {len(np.unique(subjects))}")
    
    # 只使用Session 0的数据进行训练
    train_mask = sessions == 0
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    print(f"训练数据: {X_train_all.shape}")
    
    # ==================== 训练GAN ====================
    print("\n" + "="*60)
    print("1/4: 训练GAN")
    print("="*60)
    
    gan_model = train_gan(X_train_all, y_train_all, epochs=100, batch_size=64)
    
    # 生成GAN样本
    print("\n" + "="*60)
    print("2/4: 生成GAN样本")
    print("="*60)
    
    gan_X, gan_y = generate_samples(gan_model, 'GAN', n_samples=2000)
    
    # 保存GAN样本
    os.makedirs('outputs/gan_samples', exist_ok=True)
    np.save('outputs/gan_samples/gan_samples.npy', gan_X)
    np.save('outputs/gan_samples/gan_labels.npy', gan_y)
    print("✅ GAN样本已保存")
    
    # 评估GAN
    gan_results = evaluate_method(X, y, subjects, sessions, gan_X, gan_y, 'GAN')
    
    # 保存GAN结果
    import json
    gan_output = {
        'method': 'GAN',
        'within_subject': gan_results['within_subject'],
        'cross_session': gan_results['cross_session'],
        'cross_subject': gan_results['cross_subject']
    }
    with open('outputs/results/gan_all_scenarios.json', 'w') as f:
        json.dump(gan_output, f, indent=2)
    print("\n✅ GAN结果已保存到: outputs/results/gan_all_scenarios.json")
    
    # ==================== 训练VAE ====================
    print("\n" + "="*60)
    print("3/4: 训练VAE")
    print("="*60)
    
    vae_model = train_vae(X_train_all, y_train_all, epochs=100, batch_size=64)
    
    # 生成VAE样本
    print("\n" + "="*60)
    print("4/4: 生成VAE样本")
    print("="*60)
    
    vae_X, vae_y = generate_samples(vae_model, 'VAE', n_samples=2000)
    
    # 保存VAE样本
    os.makedirs('outputs/vae_samples', exist_ok=True)
    np.save('outputs/vae_samples/vae_samples.npy', vae_X)
    np.save('outputs/vae_samples/vae_labels.npy', vae_y)
    print("✅ VAE样本已保存")
    
    # 评估VAE
    vae_results = evaluate_method(X, y, subjects, sessions, vae_X, vae_y, 'VAE')
    
    # 保存VAE结果
    vae_output = {
        'method': 'VAE',
        'within_subject': vae_results['within_subject'],
        'cross_session': vae_results['cross_session'],
        'cross_subject': vae_results['cross_subject']
    }
    with open('outputs/results/vae_all_scenarios.json', 'w') as f:
        json.dump(vae_output, f, indent=2)
    print("\n✅ VAE结果已保存到: outputs/results/vae_all_scenarios.json")
    
    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("训练和评估完成！")
    print("="*60)
    
    print("\nGAN结果:")
    for scenario in ['within_subject', 'cross_session', 'cross_subject']:
        print(f"  {scenario}: {gan_results[scenario]['mean']:.4f} ± {gan_results[scenario]['std']:.4f}")
    
    print("\nVAE结果:")
    for scenario in ['within_subject', 'cross_session', 'cross_subject']:
        print(f"  {scenario}: {vae_results[scenario]['mean']:.4f} ± {vae_results[scenario]['std']:.4f}")
    
    print("\n下一步: 运行完整对比")
    print("python experiments/paper_experiments/3_compare_all_augmentation_methods.py")

if __name__ == '__main__':
    main()
