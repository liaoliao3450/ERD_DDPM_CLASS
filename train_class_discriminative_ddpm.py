"""
Class-Discriminative DDPM 训练脚本

训练具有增强类别区分能力的EEG-DDPM模型。

主要特性:
1. 多尺度类别条件注入 (EnhancedClassEmbedding + MultiScaleCondUNet)
2. 组合损失函数 (噪声 + 频谱 + ERD侧化 + 分类)
3. 分类器引导采样
4. 综合评估框架

Requirements: 1.1, 2.1, 2.4
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 添加模块路径
sys.path.insert(0, 'core/models/ddpm')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    EvaluationMetrics, pretrain_classifier
)

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = 'data/processed/BCI2a'
CHECKPOINT_DIR = 'checkpoints'
OUTPUT_DIR = 'outputs/figures'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据参数
C, T, NUM_CLASSES = 22, 1000, 4
FS = 250
C3_IDX, C4_IDX = 7, 11

# 训练参数
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# 损失权重
# 诊断: cls_weight>0 会导致模型过度强化类别特征 (gs=0 时 C0-C2=202 vs Real 4.0)
# 先用 cls_weight=0 验证: 是否是分类损失导致类别过度分离
NOISE_WEIGHT = 1.0
SPECTRAL_WEIGHT = 1.0
ERD_WEIGHT = 2.0
CLS_WEIGHT = 0.0  # 临时设为 0，验证分类损失是否是问题根源

# 分类器引导
GUIDANCE_SCALE = 2.0
CLASSIFIER_PRETRAIN_EPOCHS = 50


# ============================================================================
# 数据处理函数
# ============================================================================

def load_data():
    """加载并预处理BCI2a数据（逐通道归一化，与BCI2b/PhysioNet一致）"""
    print("加载数据...")
    
    X = np.load(f'{DATA_DIR}/X.npy') * 1e6  # 转换为微伏
    y = np.load(f'{DATA_DIR}/y.npy')
    
    # 使用session 0作为训练数据
    sess_ids = np.tile(np.repeat([0, 1], 288), 9)
    mask = sess_ids == 0
    X_train = X[mask]
    y_train = (y[mask] - 1).astype(np.int64)  # 标签从0开始
    
    # 逐通道标准化 (与BCI2b/PhysioNet一致)
    data_mean = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # [1, C, 1]
    data_std = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)    # [1, C, 1]
    data_std = np.maximum(data_std, 1e-6)
    X_norm = ((X_train - data_mean) / data_std).astype(np.float32)
    # Clip极端值
    clip_count = np.sum(np.abs(X_norm) > 5)
    X_norm = np.clip(X_norm, -5.0, 5.0)
    
    print(f"  训练数据: {X_norm.shape}")
    print(f"  类别分布: {np.bincount(y_train)}")
    print(f"  逐通道标准化: mean={data_mean.mean():.6f}, std={data_std.mean():.6f}")
    print(f"  标准化后数据范围: [{X_norm.min():.4f}, {X_norm.max():.4f}] (clipped {clip_count} values > 5)")
    
    return X_norm, y_train, data_mean, data_std


def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd


def compute_class_laterality(X, y):
    """计算每个类别的平均侧化指数"""
    laterality = torch.zeros(NUM_CLASSES)
    
    for cls in range(NUM_CLASSES):
        cls_data = X[y == cls]
        lat_values = []
        
        for i in range(len(cls_data)):
            f, psd_c3 = signal.welch(cls_data[i, C3_IDX], fs=FS, nperseg=256)
            f, psd_c4 = signal.welch(cls_data[i, C4_IDX], fs=FS, nperseg=256)
            alpha_mask = (f >= 8) & (f <= 13)
            c3_alpha = psd_c3[alpha_mask].mean()
            c4_alpha = psd_c4[alpha_mask].mean()
            lat = (c4_alpha - c3_alpha) / (c4_alpha + c3_alpha + 1e-10)
            lat_values.append(lat)
        
        laterality[cls] = float(np.mean(lat_values))
    
    return laterality

def load_or_train_classifier(X_train, y_train, device=DEVICE, classifier_path=f'{CHECKPOINT_DIR}/classifier_class_disc.pt'):
    """
    加载预训练分类器，如果不存在则训练并保存
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        device: 设备
        classifier_path: 分类器保存路径
    
    Returns:
        classifier: 训练好的分类器
    """
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    
    # 检查是否存在预训练分类器
    if os.path.exists(classifier_path):
        print(f"  加载预训练分类器: {classifier_path}")
        try:
            checkpoint = torch.load(classifier_path, map_location=device)
            # 支持两种保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            classifier.eval()
            print("  ✅ 分类器加载成功！")
            return classifier
        except Exception as e:
            print(f"  ⚠️  加载失败 ({e})，将重新训练...")
    
    # 如果不存在或加载失败，则训练
    print(f"  预训练分类器不存在，开始训练 ({CLASSIFIER_PRETRAIN_EPOCHS} epochs)...")
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
    
    classifier = pretrain_classifier(
        classifier,
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        epochs=CLASSIFIER_PRETRAIN_EPOCHS,
        batch_size=64,
        lr=1e-3,
        device=device,
        save_path=classifier_path,
        verbose=True
    )
    
    print(f"  ✅ 分类器训练完成并已保存到: {classifier_path}")
    return classifier


# ============================================================================
# 训练函数
# ============================================================================

def train(ddpm, loader, optimizer, scheduler, epochs,
          noise_w=1.0, spectral_w=1.0, erd_w=2.0, cls_w=0.0,
          save_path='checkpoints/best_class_discriminative.pt',
          log_interval=25, data_mean=None, data_std=None):
    """
    主训练函数

    Args:
        ddpm: ClassDiscriminativeDDPM模型
        loader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        noise_w, spectral_w, erd_w, cls_w: 损失权重
        save_path: 模型保存路径
        log_interval: 日志打印间隔
        data_mean, data_std: 训练数据的逐通道均值/标准差，保存到 checkpoint

    Requirements: 1.1, 2.4
    """
    print(f"\n开始训练 ({epochs} epochs)")
    print(f"  损失权重: noise={noise_w}, spectral={spectral_w}, erd={erd_w}, cls={cls_w}")
    
    best_loss = float('inf')
    
    # 每类损失统计
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    for ep in range(1, epochs + 1):
        ddpm.train()
        
        # 损失累计
        loss_sums = {'noise': 0, 'spectral': 0, 'erd': 0, 'classification': 0, 'total': 0}
        
        # 每类损失统计 (Requirements: 2.4)
        class_loss_sums = {cls: {'erd': 0, 'count': 0} for cls in range(NUM_CLASSES)}
        
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # 计算损失
            loss, loss_dict = ddpm.loss(
                xb, yb,
                noise_weight=noise_w,
                spectral_weight=spectral_w,
                erd_weight=erd_w,
                cls_weight=cls_w
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.eps_model.parameters(), 1.0)
            optimizer.step()
            
            # 累计损失
            for key in loss_sums:
                if key in loss_dict:
                    loss_sums[key] += loss_dict[key]
            
            # 每类统计
            for cls in range(NUM_CLASSES):
                cls_mask = yb == cls
                if cls_mask.sum() > 0:
                    class_loss_sums[cls]['count'] += cls_mask.sum().item()
        
        scheduler.step()
        
        # 计算平均损失
        n = len(loader)
        avg_losses = {k: v / n for k, v in loss_sums.items()}
        
        # 打印日志
        if ep % log_interval == 0 or ep == 1:
            print(f"Epoch {ep}: noise={avg_losses['noise']:.4f}, "
                  f"spec={avg_losses['spectral']:.4f}, "
                  f"erd={avg_losses['erd']:.4f}, "
                  f"cls={avg_losses['classification']:.4f}")
            
            # 每类侧化统计 (Requirements: 2.4)
            if ep % log_interval == 0:
                print("  每类样本数:", end=" ")
                for cls in range(NUM_CLASSES):
                    print(f"{class_names[cls]}={class_loss_sums[cls]['count']}", end=" ")
                print()
        
        # 保存最佳模型
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save({
                'model_state_dict': ddpm.state_dict(),
                'target_psd': ddpm.target_psd.cpu(),
                'target_laterality': ddpm.target_laterality.cpu(),
                'epoch': ep,
                'best_loss': best_loss,
                'channels': C,
                'n_samples': T,
                'fs': FS,
                'num_classes': NUM_CLASSES,
                'data_mean': data_mean,
                'data_std': data_std,
                'data_loader': 'BCI2a',
            }, save_path)
    
    print(f"\n训练完成! 最佳损失: {best_loss:.4f}")
    print(f"模型保存到: {save_path}")
    
    return ddpm


# ============================================================================
# 评估和可视化
# ============================================================================

def evaluate_model(ddpm, X_train, y_train, n_samples=100, guidance_scale=2.0):
    """
    评估模型
    
    Args:
        ddpm: 训练好的DDPM模型
        X_train: 训练数据
        y_train: 训练标签
        n_samples: 每类生成样本数
        guidance_scale: 引导强度
        
    Returns:
        results: 评估结果字典
    """
    print("\n" + "=" * 60)
    print("评估")
    print("=" * 60)
    
    ddpm.eval()
    metrics = EvaluationMetrics(fs=FS, c3_idx=C3_IDX, c4_idx=C4_IDX)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    # 采样真实数据
    idx = np.random.choice(len(X_train), n_samples, replace=False)
    real_data = X_train[idx]
    real_labels = y_train[idx]
    
    # 生成数据
    print(f"生成数据 (guidance_scale={guidance_scale})...")
    gen_data = []
    gen_labels = []
    
    n_per_class = n_samples // NUM_CLASSES
    for cls in range(NUM_CLASSES):
        y_gen = torch.full((n_per_class,), cls, device=DEVICE, dtype=torch.long)
        data = ddpm.sample_ddim(n_per_class, y_gen, steps=50, guidance_scale=guidance_scale)
        gen_data.append(data.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
    
    gen_data = np.concatenate(gen_data)
    gen_labels = np.array(gen_labels)
    
    # 基本统计
    print("\n基本统计:")
    print(f"  真实: mean={real_data.mean():.4f}, std={real_data.std():.4f}")
    print(f"  生成: mean={gen_data.mean():.4f}, std={gen_data.std():.4f}")
    
    # 完整评估
    results = metrics.evaluate(real_data, real_labels, gen_data, gen_labels)
    
    # 打印侧化指数
    print("\nERD侧化指数:")
    target_lat = ddpm.target_laterality.cpu().numpy()
    for cls in range(NUM_CLASSES):
        real_lat = results['per_class_laterality'][cls]['real']
        gen_lat = results['per_class_laterality'][cls]['generated']
        print(f"  {class_names[cls]}: Target={target_lat[cls]:+.4f}, "
              f"Real={real_lat:+.4f}, Gen={gen_lat:+.4f}")
    
    # 打印频段功率比
    print("\n频段功率比 (Gen/Real):")
    for band, ratio in results['band_power_ratios'].items():
        status = "✓" if 0.8 <= ratio <= 1.2 else "✗"
        print(f"  {band.capitalize()}: {ratio:.2f}x {status}")
    
    # 打印分类准确率
    print("\n分类准确率:")
    print(f"  LDA (Real): {results['lda_accuracy']['real']*100:.2f}%")
    print(f"  LDA (Gen): {results['lda_accuracy']['generated']*100:.2f}%")
    print(f"  Cross-val (Real→Gen): {results['cross_val_accuracy']*100:.2f}%")
    
    return results, real_data, real_labels, gen_data, gen_labels


def visualize_results(real_data, real_labels, gen_data, gen_labels, 
                      results, save_path='outputs/figures/class_discriminative_eval.png'):
    """可视化评估结果"""
    print("\n生成可视化...")
    
    metrics = EvaluationMetrics(fs=FS)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors = ['red', 'blue', 'green', 'purple']
    
    # 提取特征
    real_feat = metrics.extract_features(real_data)
    gen_feat = metrics.extract_features(gen_data)
    
    # t-SNE
    all_feat = np.vstack([real_feat, gen_feat])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_feat)
    
    n_real = len(real_data)
    real_emb = embedded[:n_real]
    gen_emb = embedded[n_real:]
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 按类别的t-SNE
    ax = axes[0]
    for cls in range(NUM_CLASSES):
        mask_real = real_labels == cls
        mask_gen = gen_labels == cls
        ax.scatter(real_emb[mask_real, 0], real_emb[mask_real, 1],
                   c=colors[cls], marker='o', alpha=0.6, s=40, label=f'{class_names[cls]} (Real)')
        ax.scatter(gen_emb[mask_gen, 0], gen_emb[mask_gen, 1],
                   c=colors[cls], marker='x', alpha=0.6, s=40)
    ax.set_title('t-SNE by Class (Class-Discriminative DDPM)')
    ax.legend(loc='upper right', fontsize=8)
    
    # 2. Real vs Generated
    ax = axes[1]
    ax.scatter(real_emb[:, 0], real_emb[:, 1], c='blue', marker='o', alpha=0.6, label='Real', s=50)
    ax.scatter(gen_emb[:, 0], gen_emb[:, 1], c='red', marker='x', alpha=0.6, label='Generated', s=50)
    ax.set_title('t-SNE: Real vs Generated')
    ax.legend()
    
    # 3. 分类准确率柱状图
    ax = axes[2]
    methods = ['LDA\n(Real)', 'LDA\n(Gen)', 'Cross-val\n(Real→Gen)']
    scores = [
        results['lda_accuracy']['real'] * 100,
        results['lda_accuracy']['generated'] * 100,
        results['cross_val_accuracy'] * 100
    ]
    bars = ax.bar(methods, scores, color=['blue', 'red', 'green'], alpha=0.7)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Class Separability')
    ax.set_ylim(0, 100)
    ax.axhline(25, color='gray', linestyle='--', label='Random (25%)')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("Class-Discriminative DDPM 训练")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"\n设备: {DEVICE}")
    
    # 加载数据
    X_norm, y_train, data_mean, data_std = load_data()
    
    # 计算目标统计量
    print("\n计算目标统计量...")
    target_psd = compute_target_psd(X_norm).to(DEVICE)
    target_laterality = compute_class_laterality(X_norm, y_train).to(DEVICE)
    
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    print("\n目标侧化指数:")
    for cls in range(NUM_CLASSES):
        print(f"  {class_names[cls]}: {target_laterality[cls].item():+.4f}")
    
    # 创建模型
    print("\n创建模型...")
    eps_model = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(DEVICE)
    
    n_params_eps = sum(p.numel() for p in eps_model.parameters()) / 1e6
    print(f"  UNet参数量: {n_params_eps:.2f}M")
    
    # 加载或训练分类器（如果已存在预训练模型则直接加载）
    print(f"\n加载/训练分类器...")
    classifier = load_or_train_classifier(X_norm, y_train, device=DEVICE)
    
    n_params_cls = sum(p.numel() for p in classifier.parameters()) / 1e3
    print(f"  分类器参数量: {n_params_cls:.2f}K (EEGNet)")
    
    # 创建DDPM
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=FS
    ).to(DEVICE)
    
    # 创建数据加载器
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_norm), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(eps_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    # 训练
    ddpm = train(
        ddpm, loader, optimizer, scheduler,
        epochs=EPOCHS,
        noise_w=NOISE_WEIGHT,
        spectral_w=SPECTRAL_WEIGHT,
        erd_w=ERD_WEIGHT,
        cls_w=CLS_WEIGHT,
        save_path=f'{CHECKPOINT_DIR}/best_class_discriminative.pt',
        log_interval=25,
        data_mean=data_mean,
        data_std=data_std
    )
    
    # 加载最佳模型
    best_ckpt = torch.load(f'{CHECKPOINT_DIR}/best_class_discriminative.pt', map_location=DEVICE, weights_only=False)
    if isinstance(best_ckpt, dict) and 'model_state_dict' in best_ckpt:
        ddpm.load_state_dict(best_ckpt['model_state_dict'])
    else:
        ddpm.load_state_dict(best_ckpt)
    
    # 评估
    results, real_data, real_labels, gen_data, gen_labels = evaluate_model(
        ddpm, X_norm, y_train,
        n_samples=100,
        guidance_scale=GUIDANCE_SCALE
    )
    
    # 可视化
    visualize_results(
        real_data, real_labels, gen_data, gen_labels, results,
        save_path=f'{OUTPUT_DIR}/class_discriminative_eval.png'
    )
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
