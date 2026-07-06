"""
重新生成GAN和VAE样本用于对比实验
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
from pathlib import Path

# 添加模型路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'gan'))
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'vae'))

from model import Gen1D
from vae_model import VAE1D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_gan_model():
    """加载GAN模型"""
    print("\n加载GAN模型...")
    ckpt_path = 'checkpoints/gan/gan_epoch_20.pt'
    
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 未找到GAN checkpoint: {ckpt_path}")
        return None
    
    try:
        G = Gen1D(z_dim=128, out_channels=22, out_length=1000, num_classes=4, cond_embed_dim=32).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(state['G'])
        G.eval()
        print(f"  ✅ GAN加载成功")
        return G
    except Exception as e:
        print(f"  ❌ GAN加载失败: {e}")
        return None

def load_vae_model():
    """加载VAE模型"""
    print("\n加载VAE模型...")
    ckpt_path = 'checkpoints/vae/vae_epoch_20.pt'
    
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 未找到VAE checkpoint: {ckpt_path}")
        return None
    
    try:
        vae = VAE1D(in_channels=22, latent_dim=128, num_classes=4).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        vae.load_state_dict(state['model'])
        vae.eval()
        print(f"  ✅ VAE加载成功")
        return vae
    except Exception as e:
        print(f"  ❌ VAE加载失败: {e}")
        return None

def generate_gan_samples(G, n_samples=1000):
    """生成GAN样本"""
    print(f"\n生成GAN样本 (n={n_samples})...")
    
    gen_X = []
    gen_y = []
    
    batch_size = 50
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - len(gen_X))
            
            # 为每个类别生成样本
            for c in range(4):
                n_per_class = current_batch_size // 4
                if c < current_batch_size % 4:
                    n_per_class += 1
                
                if n_per_class > 0:
                    z = torch.randn(n_per_class, 128, device=DEVICE)
                    y = torch.full((n_per_class,), c, dtype=torch.long, device=DEVICE)
                    samples = G(z, y)
                    
                    gen_X.append(samples.cpu().numpy())
                    gen_y.extend([c] * n_per_class)
            
            if (i + 1) % 5 == 0:
                print(f"  进度: {len(gen_X)}/{n_samples}")
    
    gen_X = np.concatenate(gen_X, axis=0)[:n_samples]
    gen_y = np.array(gen_y[:n_samples])
    
    print(f"  生成完成: {gen_X.shape}")
    print(f"  数据范围: [{gen_X.min():.3f}, {gen_X.max():.3f}]")
    print(f"  均值: {gen_X.mean():.3f}, 标准差: {gen_X.std():.3f}")
    
    return gen_X, gen_y

def generate_vae_samples(vae, n_samples=1000):
    """生成VAE样本"""
    print(f"\n生成VAE样本 (n={n_samples})...")
    
    gen_X = []
    gen_y = []
    
    batch_size = 50
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - len(gen_X))
            
            # 为每个类别生成样本
            for c in range(4):
                n_per_class = current_batch_size // 4
                if c < current_batch_size % 4:
                    n_per_class += 1
                
                if n_per_class > 0:
                    z = torch.randn(n_per_class, 128, device=DEVICE)
                    y = torch.full((n_per_class,), c, dtype=torch.long, device=DEVICE)
                    samples = vae.decode(z, y)
                    
                    gen_X.append(samples.cpu().numpy())
                    gen_y.extend([c] * n_per_class)
            
            if (i + 1) % 5 == 0:
                print(f"  进度: {len(gen_X)}/{n_samples}")
    
    gen_X = np.concatenate(gen_X, axis=0)[:n_samples]
    gen_y = np.array(gen_y[:n_samples])
    
    print(f"  生成完成: {gen_X.shape}")
    print(f"  数据范围: [{gen_X.min():.3f}, {gen_X.max():.3f}]")
    print(f"  均值: {gen_X.mean():.3f}, 标准差: {gen_X.std():.3f}")
    
    return gen_X, gen_y

def main():
    print("="*60)
    print("重新生成GAN和VAE样本")
    print("="*60)
    print(f"设备: {DEVICE}")
    
    # 生成GAN样本
    gan_model = load_gan_model()
    if gan_model is not None:
        gan_X, gan_y = generate_gan_samples(gan_model, n_samples=2000)
        
        # 保存
        os.makedirs('outputs/gan_samples', exist_ok=True)
        np.save('outputs/gan_samples/gan_samples.npy', gan_X)
        np.save('outputs/gan_samples/gan_labels.npy', gan_y)
        print(f"\n✅ GAN样本已保存到: outputs/gan_samples/")
    else:
        print("\n❌ GAN样本生成失败")
    
    # 生成VAE样本
    vae_model = load_vae_model()
    if vae_model is not None:
        vae_X, vae_y = generate_vae_samples(vae_model, n_samples=2000)
        
        # 保存
        os.makedirs('outputs/vae_samples', exist_ok=True)
        np.save('outputs/vae_samples/vae_samples.npy', vae_X)
        np.save('outputs/vae_samples/vae_labels.npy', vae_y)
        print(f"\n✅ VAE样本已保存到: outputs/vae_samples/")
    else:
        print("\n❌ VAE样本生成失败")
    
    print("\n" + "="*60)
    print("样本生成完成！")
    print("="*60)
    print("\n下一步: 运行对比实验")
    print("python experiments/paper_experiments/3_compare_generative_models_complete.py")

if __name__ == '__main__':
    main()
