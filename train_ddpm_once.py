#!/usr/bin/env python3
"""
训练DDPM并保存（只需运行一次）

训练完整的Class-Discriminative DDPM并保存，后续实验直接加载
"""
import sys, os, torch, numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM
)
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    """加载数据"""
    X, y, subjects, sessions = load_bci2a_data()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def compute_targets(X, y):
    """计算目标特征"""
    C, T, NC, FS = 22, 1000, 4, 250
    psd = np.mean([np.abs(np.fft.rfft(X[i]))**2 for i in range(len(X))], axis=(0,1))
    lat = []
    for c in range(NC):
        m = y == c
        if m.sum() == 0: 
            lat.append(0.0)
            continue
        d = X[m]
        f = np.fft.rfftfreq(T, 1.0/FS)
        am = (f >= 8) & (f <= 13)
        c3 = np.abs(np.fft.rfft(d[:,7,:])[:,am])**2
        c4 = np.abs(np.fft.rfft(d[:,11,:])[:,am])**2
        lat.append(float((c4.mean() - c3.mean()) / (c4.mean() + c3.mean() + 1e-10)))
    return torch.tensor(psd, dtype=torch.float32), torch.tensor(lat, dtype=torch.float32)

def main():
    print("="*70)
    print("🎨 训练Class-Discriminative DDPM（只需运行一次）")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    print("📁 加载数据...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 计算目标
    print("\n🎯 计算目标特征...")
    tpsd, tlat = compute_targets(X_train, y_train)
    
    # 初始化模型
    print("\n🏗️  初始化模型...")
    eps = MultiScaleCondUNet(channels=22, num_classes=4).to(DEVICE)
    
    # 加载预训练分类器
    print("\n📥 加载预训练分类器...")
    clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
    checkpoint = torch.load('checkpoints/pretrained_classifier.pt', map_location=DEVICE)
    clf.load_state_dict(checkpoint['model_state_dict'])
    print("  ✅ 分类器加载成功")
    
    # 创建DDPM
    ddpm = ClassDiscriminativeDDPM(eps, clf, tpsd.to(DEVICE), tlat.to(DEVICE), 1000, 22, 1000).to(DEVICE)
    
    # 训练DDPM
    print("\n🎨 训练DDPM (500 epochs)...")
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=32, shuffle=True, drop_last=True
    )
    opt = torch.optim.AdamW(ddpm.eps_model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    
    best_loss = float('inf')
    best_epoch = 0
    
    for e in range(500):
        ddpm.train()
        epoch_loss = 0
        n_batches = 0
        for x, yb in loader:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            # 使用优化的权重
            loss, _ = ddpm.loss(x, yb, erd_weight=0.5, cls_weight=1.0,
                               noise_weight=1.0, spectral_weight=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = e + 1
        
        if (e+1) % 10 == 0:
            print(f"  Epoch {e+1}/500, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (Epoch {best_epoch})")
    
    print(f"\n✅ 训练完成！最佳Loss: {best_loss:.4f} (Epoch {best_epoch})")
    
    # 保存DDPM
    print("\n💾 保存DDPM模型...")
    os.makedirs('checkpoints', exist_ok=True)
    
    checkpoint = {
        'epoch': 500,
        'model_state_dict': ddpm.state_dict(),
        'eps_model_state_dict': ddpm.eps_model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'best_loss': best_loss,
        'target_psd': tpsd,
        'target_laterality': tlat,
    }
    
    torch.save(checkpoint, 'checkpoints/trained_ddpm.pt')
    print("  ✅ 完整模型: checkpoints/trained_ddpm.pt")
    
    # 只保存eps_model权重（更小）
    torch.save(ddpm.eps_model.state_dict(), 'checkpoints/trained_ddpm_eps_only.pt')
    print("  ✅ Eps模型: checkpoints/trained_ddpm_eps_only.pt")
    
    # 保存配置
    import json
    config = {
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
        'timesteps': 1000,
        'erd_weight': 0.5,
        'cls_weight': 1.0,
        'noise_weight': 1.0,
        'spectral_weight': 0.5,
        'guidance_scale': 3.0,
        'ddim_steps': 50,
    }
    
    with open('checkpoints/ddpm_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("  ✅ 配置文件: checkpoints/ddpm_config.json")
    
    # 测试生成
    print("\n🎲 测试生成...")
    ddpm.eval()
    with torch.no_grad():
        y_test = torch.tensor([0, 1, 2, 3], device=DEVICE)
        samples = ddpm.sample_ddim(4, y_test, steps=50, guidance_scale=3.0)
        print(f"  ✅ 成功生成 {len(samples)} 个样本")
        print(f"  样本形状: {samples.shape}")
    
    print("\n" + "="*70)
    print("✅ DDPM训练并保存完成！")
    print("="*70)
    print("\n保存的文件:")
    print("  1. checkpoints/trained_ddpm.pt (完整模型)")
    print("  2. checkpoints/trained_ddpm_eps_only.pt (仅eps模型)")
    print("  3. checkpoints/ddpm_config.json (配置)")
    print("\n后续实验可以直接加载这个DDPM，无需重复训练！")

if __name__ == '__main__':
    main()
