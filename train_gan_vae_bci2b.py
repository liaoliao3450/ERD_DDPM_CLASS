#!/usr/bin/env python3
"""
BCI2b GAN / VAE 训练脚本（单独版本，不覆盖 BCI2a）

目标：
- 使用与 BCI2a 相同结构的 Gen1D / Disc1D / VAE1D
- 但通道数与类别数改为 BCI2b（C=3, num_classes=2）
- 只用 session==0 (T) 作为生成模型训练集（与其他 BCI2b 脚本一致）

输出（全部放在 checkpoints/bci2b/ 下，与 BCI2a 分离）：
- checkpoints/bci2b/gan_bci2b.pt   （{'G': state_dict, 'D': state_dict}）
- checkpoints/bci2b/vae_bci2b.pt   （{'model': state_dict}）
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 数据加载
sys.path.insert(0, str(PROJECT_ROOT / "utils"))
from data_loader_bci2b import load_bci2b_data  # type: ignore

# GAN 模型
sys.path.insert(0, str(PROJECT_ROOT / "core" / "models" / "gan"))
from model import Gen1D, Disc1D  # type: ignore

# VAE 模型
sys.path.insert(0, str(PROJECT_ROOT / "core" / "models" / "vae"))
from vae_model import VAE1D  # type: ignore


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_gan_bci2b(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 500,
    batch_size: int = 64,
) -> Tuple[Gen1D, Disc1D]:
    """按 BCI2a 脚本风格训练条件 GAN，但改为 BCI2b 维度。"""
    print("\n开始训练 BCI2b GAN ...")
    print(f"  训练数据形状: {X.shape}, 轮数: {epochs}")

    n_samples, C, T = X.shape
    num_classes = int(len(np.unique(y)))

    G = Gen1D(z_dim=128, out_channels=C, out_length=T, num_classes=num_classes, cond_embed_dim=32).to(DEVICE)
    D = Disc1D(in_channels=C, in_length=T, num_classes=num_classes, cond_embed_dim=32).to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    for epoch in range(epochs):
        G.train()
        D.train()
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            real_X = X_tensor[batch_indices].to(DEVICE)
            real_y = y_tensor[batch_indices].to(DEVICE)
            bs = real_X.size(0)

            # 判别器
            opt_D.zero_grad()
            real_pred = D(real_X, real_y)
            loss_D_real = -torch.mean(real_pred)

            z = torch.randn(bs, 128, device=DEVICE)
            fake_X = G(z, real_y)
            fake_pred = D(fake_X.detach(), real_y)
            loss_D_fake = torch.mean(fake_pred)

            alpha = torch.rand(bs, 1, 1, device=DEVICE)
            interpolates = (alpha * real_X + (1.0 - alpha) * fake_X.detach()).requires_grad_(True)
            d_interpolates = D(interpolates, real_y)
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True,
            )[0]
            gradient_penalty = ((gradients.view(bs, -1).norm(2, dim=1) - 1.0) ** 2).mean() * 10.0

            loss_D = loss_D_real + loss_D_fake + gradient_penalty
            loss_D.backward()
            opt_D.step()

            # 生成器（按原脚本，每若干步更新一次，此处简单每个 batch 都更新）
            opt_G.zero_grad()
            z = torch.randn(bs, 128, device=DEVICE)
            fake_X = G(z, real_y)
            fake_pred = D(fake_X, real_y)
            loss_G = -torch.mean(fake_pred)
            loss_G.backward()
            opt_G.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}")

    print("BCI2b GAN 训练完成")
    return G, D


def train_vae_bci2b(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 500,
    batch_size: int = 64,
) -> VAE1D:
    """按 BCI2a 脚本风格训练条件 VAE，但改为 BCI2b 维度。"""
    print("\n开始训练 BCI2b VAE ...")
    print(f"  训练数据形状: {X.shape}, 轮数: {epochs}")

    n_samples, C, T = X.shape
    num_classes = int(len(np.unique(y)))

    vae = VAE1D(channels=C, length=T, latent_dim=128, cond_dim=32, num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    for epoch in range(epochs):
        vae.train()
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_X = X_tensor[batch_indices].to(DEVICE)
            batch_y = y_tensor[batch_indices].to(DEVICE)

            optimizer.zero_grad()
            recon_X, mu, logvar = vae(batch_X, batch_y)
            recon_loss = nn.functional.mse_loss(recon_X, batch_X, reduction="sum") / batch_X.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}")

    print("BCI2b VAE 训练完成")
    return vae


def main() -> None:
    print("=" * 70)
    print("BCI2b GAN / VAE 训练（与 BCI2a 结构一致，参数适配二分类）")
    print("=" * 70)
    print(f"设备: {DEVICE}\n")

    # 加载 BCI2b 数据（标准化与其他脚本保持一致）
    X, y, subjects, sessions, subj_map = load_bci2b_data()
    print(f"总数据: X={X.shape}, y={y.shape}, 被试数={len(subj_map)}")

    # 仅使用 session==0 (T) 训练生成模型
    train_mask = sessions == 0
    if not train_mask.any():
        raise RuntimeError("BCI2b 数据中不存在 session==0 (T) 的样本，无法训练 GAN/VAE。")

    X_train = X[train_mask]
    y_train = y[train_mask]
    print(f"用于 GAN/VAE 训练的数据: X_train={X_train.shape}, y_train={y_train.shape}")

    os.makedirs("checkpoints/bci2b", exist_ok=True)

    # 训练 GAN
    G, D = train_gan_bci2b(X_train, y_train, epochs=500, batch_size=64)
    gan_ckpt_path = "checkpoints/bci2b/gan_bci2b.pt"
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, gan_ckpt_path)
    print(f"\nGAN 权重已保存到: {gan_ckpt_path}")

    # 训练 VAE
    vae = train_vae_bci2b(X_train, y_train, epochs=500, batch_size=64)
    vae_ckpt_path = "checkpoints/bci2b/vae_bci2b.pt"
    torch.save({"model": vae.state_dict()}, vae_ckpt_path)
    print(f"\nVAE 权重已保存到: {vae_ckpt_path}")

    print("\n" + "=" * 70)
    print("BCI2b GAN / VAE 训练完成（不会覆盖 BCI2a 权重）")
    print("=" * 70)


if __name__ == "__main__":
    main()



