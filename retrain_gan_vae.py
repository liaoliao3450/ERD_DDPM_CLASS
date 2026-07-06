#!/usr/bin/env python3
"""
重新训练GAN和VAE，使用修复后的模型（带输出缩放）
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '.')
from utils.data_loader import load_bci2a_data, get_subject_session_data
from core.models.timegan.model import TimeGAN
from core.models.latent_diffusion.model import LatentDiffusion1D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_gan(X_train, y_train, epochs=300, batch_size=32):
    """训练TimeGAN"""
    print("\n" + "="*70)
    print("Training TimeGAN with Output Scaling")
    print("="*70)
    
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    
    # 创建模型
    model = TimeGAN(channels=n_channels, length=n_samples, hidden=64, z_dim=64).to(DEVICE)
    
    # 优化器
    opt_ae = optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=1e-3)
    opt_g = optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=1e-3)
    opt_d = optim.Adam(model.discriminator.parameters(), lr=1e-3)
    
    # 数据加载器
    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss_ae = 0
        total_loss_g = 0
        total_loss_d = 0
        
        for batch_idx, (x_real,) in enumerate(dataloader):
            x_real = x_real.to(DEVICE)
            batch_size_actual = x_real.size(0)
            
            # 1. 训练Autoencoder (Embedder + Recovery)
            opt_ae.zero_grad()
            h_real = model.embedder(x_real)
            x_rec = model.recovery(h_real)
            loss_ae = model.reconstruction_loss(x_real, x_rec)
            loss_ae.backward()
            opt_ae.step()
            total_loss_ae += loss_ae.item()
            
            # 2. 训练Generator
            opt_g.zero_grad()
            z = torch.randn(batch_size_actual, model.hidden, model.length, device=DEVICE)
            h_fake = model.generator(z)
            h_fake_sup = model.supervisor(h_fake)
            
            d_fake = model.discriminator(h_fake)
            d_fake_sup = model.discriminator(h_fake_sup)
            
            loss_g = model.generator_loss(d_fake, d_fake_sup)
            loss_g += model.supervised_loss(h_fake, h_fake_sup)
            loss_g.backward()
            opt_g.step()
            total_loss_g += loss_g.item()
            
            # 3. 训练Discriminator
            opt_d.zero_grad()
            h_real = model.embedder(x_real).detach()
            h_fake = model.generator(z).detach()
            h_fake_sup = model.supervisor(h_fake).detach()
            
            d_real = model.discriminator(h_real)
            d_fake = model.discriminator(h_fake)
            d_fake_sup = model.discriminator(h_fake_sup)
            
            loss_d = model.discriminator_loss(d_real, d_fake, d_fake_sup)
            loss_d.backward()
            opt_d.step()
            total_loss_d += loss_d.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"AE={total_loss_ae/len(dataloader):.4f}, "
                  f"G={total_loss_g/len(dataloader):.4f}, "
                  f"D={total_loss_d/len(dataloader):.4f}, "
                  f"Scale={model.recovery.output_scale.item():.4f}")
    
    # 保存模型
    os.makedirs('checkpoints/gan', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/gan/gan_fixed_scale.pt')
    print(f"\n✅ GAN训练完成！输出缩放: {model.recovery.output_scale.item():.4f}")
    
    return model


def train_vae(X_train, y_train, epochs=300, batch_size=32):
    """训练VAE"""
    print("\n" + "="*70)
    print("Training VAE with Output Scaling")
    print("="*70)
    
    n_channels = X_train.shape[1]
    
    # 创建模型
    model = LatentDiffusion1D(channels=n_channels, z_channels=64, num_classes=4).to(DEVICE)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 数据加载器
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_real, y_batch in dataloader:
            x_real = x_real.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Encode
            z_mu, z_logvar = model.enc(x_real)
            
            # Reparameterization
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mu + eps * std
            
            # Decode
            x_rec = model.dec(z)
            
            # Loss
            recon_loss = nn.functional.mse_loss(x_rec, x_real)
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            kl_loss = kl_loss / x_real.size(0)
            
            loss = recon_loss + 0.001 * kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss={total_loss/len(dataloader):.4f}, "
                  f"Recon={recon_loss.item():.4f}, "
                  f"KL={kl_loss.item():.4f}, "
                  f"Scale={model.dec.output_scale.item():.4f}")
    
    # 保存模型
    os.makedirs('checkpoints/vae', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vae/vae_fixed_scale.pt')
    print(f"\n✅ VAE训练完成！输出缩放: {model.dec.output_scale.item():.4f}")
    
    return model


def test_generation(model, model_type, n_samples=72, target_std=0.796):
    """测试生成数据的统计特性"""
    print(f"\n测试{model_type}生成数据...")
    model.eval()
    
    generated_samples = []
    
    with torch.no_grad():
        for class_id in range(4):
            if model_type == 'GAN':
                # GAN生成
                z = torch.randn(n_samples, model.hidden, model.length, device=DEVICE)
                h_fake = model.generator(z)
                x_gen = model.recovery(h_fake)
            else:  # VAE
                # VAE生成
                z = torch.randn(n_samples, 64, device=DEVICE)
                # 需要reshape z到正确的形状
                z = z.unsqueeze(-1).repeat(1, 1, 125)  # (batch, 64, 125)
                x_gen = model.dec(z)
            
            generated_samples.append(x_gen.cpu().numpy())
    
    generated_samples = np.concatenate(generated_samples, axis=0)
    
    print(f"  生成数据形状: {generated_samples.shape}")
    print(f"  均值: {generated_samples.mean():.6f}")
    print(f"  标准差: {generated_samples.std():.6f}")
    print(f"  最小值: {generated_samples.min():.6f}")
    print(f"  最大值: {generated_samples.max():.6f}")
    
    # 手动调整output_scale以匹配目标std
    current_std = generated_samples.std()
    scale_factor = target_std / current_std
    print(f"\n  需要的缩放因子: {scale_factor:.4f}")
    print(f"  建议的output_scale: {model.recovery.output_scale.item() * scale_factor:.4f}" if model_type == 'GAN' 
          else f"  建议的output_scale: {model.dec.output_scale.item() * scale_factor:.4f}")
    
    # 应用缩放
    generated_samples_scaled = generated_samples * scale_factor
    print(f"\n  缩放后标准差: {generated_samples_scaled.std():.6f}")
    
    return generated_samples_scaled, scale_factor


def main():
    print("="*70)
    print("重新训练GAN和VAE（带输出缩放）")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    X, y, subjects, sessions = load_bci2a_data()
    X_train, y_train = get_subject_session_data(X, y, subjects, sessions, 0, 0)  # A01, Session E
    
    print(f"训练数据: {X_train.shape}")
    print(f"数据统计:")
    print(f"  均值: {X_train.mean():.6f}")
    print(f"  标准差: {X_train.std():.6f}")
    print(f"  范围: [{X_train.min():.6f}, {X_train.max():.6f}]")
    
    target_std = X_train.std()
    print(f"\n目标标准差: {target_std:.6f}")
    
    # 训练GAN
    gan_model = train_gan(X_train, y_train, epochs=300)
    gan_samples, gan_scale = test_generation(gan_model, 'GAN', target_std=target_std)
    
    # 更新GAN的output_scale
    with torch.no_grad():
        gan_model.recovery.output_scale.mul_(gan_scale)
    torch.save(gan_model.state_dict(), 'checkpoints/gan/gan_fixed_scale.pt')
    print(f"\n✅ GAN output_scale已更新为: {gan_model.recovery.output_scale.item():.4f}")
    
    # 训练VAE
    vae_model = train_vae(X_train, y_train, epochs=300)
    vae_samples, vae_scale = test_generation(vae_model, 'VAE', target_std=target_std)
    
    # 更新VAE的output_scale
    with torch.no_grad():
        vae_model.dec.output_scale.mul_(vae_scale)
    torch.save(vae_model.state_dict(), 'checkpoints/vae/vae_fixed_scale.pt')
    print(f"\n✅ VAE output_scale已更新为: {vae_model.dec.output_scale.item():.4f}")
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    print("\n模型保存在:")
    print("  - checkpoints/gan/gan_fixed_scale.pt")
    print("  - checkpoints/vae/vae_fixed_scale.pt")
    print("\n最终输出缩放:")
    print(f"  GAN: {gan_model.recovery.output_scale.item():.4f}")
    print(f"  VAE: {vae_model.dec.output_scale.item():.4f}")
    print("\n现在可以重新生成数据并可视化了！")


if __name__ == '__main__':
    main()
