"""
Class-Discriminative DDPM 评估脚本

独立评估已训练的模型，方便针对性调试
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 添加模块路径
sys.path.insert(0, 'core/models/ddpm')
from class_discriminative import (
    MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM,
    EvaluationMetrics
)

# 导入数据加载工具
sys.path.insert(0, 'utils')
from data_loader import load_bci2a_data, get_subject_session_data

# 导入其他模型
import importlib.util

# Vanilla DDPM
sys.path.insert(0, 'core/models/ddpm')
from model import UNet1D

# GAN - 使用importlib避免路径冲突
gan_spec = importlib.util.spec_from_file_location("gan_model", "core/models/gan/model.py")
gan_module = importlib.util.module_from_spec(gan_spec)
gan_spec.loader.exec_module(gan_module)
Gen1D = gan_module.Gen1D

# VAE
sys.path.insert(0, 'core/models/vae')
from vae_model import VAE1D

# ============================================================================
# 配置
# ============================================================================

DATA_DIR = 'data/processed/BCI2a'
CHECKPOINT_DIR = 'checkpoints'
OUTPUT_DIR = 'outputs/figures'
CACHE_DIR = 'outputs/cache'  # 缓存目录
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据参数
C, T, NUM_CLASSES = 22, 1000, 4
FS = 250
C3_IDX, C4_IDX = 7, 11

# 评估参数
N_SAMPLES = 288  # 每类生成样本数
GUIDANCE_SCALE = 0.5  # 引导强度（降低以改善分布一致性）
DDIM_STEPS = 100  # DDIM采样步数（增加以提高质量）
USE_DDIM = False  # 是否使用DDIM采样（False使用标准采样，可能更接近训练分布）
USE_CLASSIFIER_FEATURES = True  # 使用分类器中间层特征（提升类别判别性）
MATCH_STATISTICS = False  # 是否对生成数据进行统计特性匹配（暂时禁用以观察引导强度效果）
MATCH_STRENGTH = 0.3  # 统计特性匹配强度（0-1，仅在MATCH_STATISTICS=True时有效）

# ============================================================================
# 数据加载函数
# ============================================================================

def load_data():
    """
    加载第一个被试第一个会话的数据
    使用与训练时相同的标准化方式（所有session 0数据的统计量）
    这确保生成数据和真实数据在相同的标准化空间中
    """
    print("加载数据...")
    print("  目标: 第一个被试 (subject_id=0) 第一个会话 (session_id=0)")
    print("  重要: 使用与训练时相同的标准化参数（所有session 0数据的统计量）")
    
    # 加载原始数据（未标准化），与训练脚本完全一致
    DATA_DIR = 'data/processed/BCI2a'
    X_raw = np.load(f'{DATA_DIR}/X.npy') * 1e6  # 转换为微伏，与训练脚本一致
    y_raw = np.load(f'{DATA_DIR}/y.npy')
    
    # 使用与训练脚本相同的标准化方式：所有session 0的数据
    sess_ids = np.tile(np.repeat([0, 1], 288), 9)
    mask_all_session0 = sess_ids == 0
    X_all_session0 = X_raw[mask_all_session0]
    
    # 计算所有session 0数据的统计量（与训练时一致）
    X_mean_all = X_all_session0.mean()
    X_std_all = X_all_session0.std()
    
    print(f"  全局统计量（所有session 0数据）:")
    print(f"    均值: {X_mean_all:.4f}, 标准差: {X_std_all:.4f}")
    
    # 获取第一个被试第一个会话的数据索引
    # 每个被试每个会话有288个样本，第一个被试第一个会话是前288个session 0的样本
    # session 0的数据索引：0-287 (被试0), 576-863 (被试1), ...
    # 第一个被试第一个会话：0-287
    n_trials_per_session = 288
    start_idx = 0
    end_idx = n_trials_per_session
    
    X_train_raw = X_all_session0[start_idx:end_idx]
    y_train = (y_raw[mask_all_session0][start_idx:end_idx] - 1).astype(np.int64)  # 标签从0开始
    
    # 使用与训练时相同的标准化参数（关键！）
    X_norm = (X_train_raw - X_mean_all) / (X_std_all )
    
    print(f"  第一个被试第一个会话数据:")
    print(f"    数据形状: {X_norm.shape}")
    print(f"    类别分布: {np.bincount(y_train)}")
    print(f"    标准化后均值: {X_norm.mean():.6f}, 标准差: {X_norm.std():.6f}")
    
    return X_norm, y_train, X_mean_all, X_std_all

def compute_target_psd(X):
    """计算目标功率谱密度"""
    fft = torch.fft.rfft(torch.FloatTensor(X), dim=-1)
    psd = (fft.abs() ** 2).mean(dim=(0, 1))
    return psd

def compute_class_laterality(X, y):
    """计算每个类别的平均侧化指数"""
    from scipy import signal
    
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

# ============================================================================
# 模型加载函数
# ============================================================================

def load_pretrained_classifier(device=DEVICE, classifier_path=f'{CHECKPOINT_DIR}/classifier_class_disc.pt'):
    """加载预训练的分类器"""
    if not os.path.exists(classifier_path):
        print(f"[错误] 分类器文件不存在: {classifier_path}")
        return None, False
    
    classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(classifier_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier.eval()
    print(f"[成功] 分类器加载成功: {classifier_path}")
    return classifier, True

def load_trained_ddpm(X_train, y_train, device=DEVICE, 
                     checkpoint_path=f'{CHECKPOINT_DIR}/best_class_discriminative.pt'):
    """加载已训练的DDPM模型"""
    if not os.path.exists(checkpoint_path):
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        return None
    
    print(f"[加载] 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 计算目标统计量
    target_psd = compute_target_psd(X_train).to(device)
    target_laterality = compute_class_laterality(X_train, y_train).to(device)
    
    # 创建模型组件
    eps_model = MultiScaleCondUNet(channels=C, num_classes=NUM_CLASSES).to(device)
    
    # 加载分类器
    classifier, loaded = load_pretrained_classifier(device)
    if not loaded:
        print("[警告] 分类器加载失败，使用随机初始化的分类器")
        classifier = EEGClassifier(channels=C, n_samples=T, num_classes=NUM_CLASSES).to(device)
    
    # 创建DDPM模型
    ddpm = ClassDiscriminativeDDPM(
        eps_model=eps_model,
        classifier=classifier,
        target_psd=target_psd,
        target_laterality=target_laterality,
        n_timesteps=1000,
        channels=C,
        n_samples=T,
        fs=FS
    ).to(device)
    
    # 加载权重
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        ddpm.load_state_dict(checkpoint['model'])
    elif isinstance(checkpoint, dict) and 'eps_model' in checkpoint:
        ddpm.eps_model.load_state_dict(checkpoint['eps_model'])
    else:
        ddpm.load_state_dict(checkpoint)
    
    ddpm.eval()
    print("[成功] 模型加载成功")
    return ddpm

def load_vanilla_ddpm(device=DEVICE, checkpoint_path=f'{CHECKPOINT_DIR}/best_ddpm.pt'):
    """加载Vanilla DDPM模型"""
    if not os.path.exists(checkpoint_path):
        print(f"[警告] Vanilla DDPM模型文件不存在: {checkpoint_path}")
        return None
    
    print(f"[加载] 加载Vanilla DDPM: {checkpoint_path}")
    try:
        model = UNet1D(in_channels=C, num_classes=NUM_CLASSES).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        print("[成功] Vanilla DDPM加载成功")
        return model
    except Exception as e:
        print(f"[错误] Vanilla DDPM加载失败: {e}")
        return None

def load_gan_model(device=DEVICE, checkpoint_path=f'{CHECKPOINT_DIR}/gan/gan_retrained.pt'):
    """加载GAN模型（使用与三种场景测试相同的模型）"""
    if not os.path.exists(checkpoint_path):
        print(f"[警告] GAN模型文件不存在: {checkpoint_path}")
        return None
    
    print(f"[加载] 加载GAN: {checkpoint_path}")
    try:
        # 使用与评估脚本相同的模型结构
        model = Gen1D(z_dim=128, out_channels=C, out_length=T, num_classes=NUM_CLASSES, cond_embed_dim=32).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'G' in checkpoint:
            model.load_state_dict(checkpoint['G'], strict=False)
        elif isinstance(checkpoint, dict) and 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        print("[成功] GAN加载成功")
        return model
    except Exception as e:
        print(f"[错误] GAN加载失败: {e}")
        return None

def load_vae_model(device=DEVICE, checkpoint_path=f'{CHECKPOINT_DIR}/vae/vae_retrained.pt'):
    """加载VAE模型（使用与三种场景测试相同的模型）"""
    if not os.path.exists(checkpoint_path):
        print(f"[警告] VAE模型文件不存在: {checkpoint_path}")
        return None
    
    print(f"[加载] 加载VAE: {checkpoint_path}")
    try:
        # 使用与评估脚本相同的模型结构
        model = VAE1D(channels=C, length=T, latent_dim=128, cond_dim=32, num_classes=NUM_CLASSES).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("[成功] VAE加载成功")
        return model
    except Exception as e:
        print(f"[错误] VAE加载失败: {e}")
        return None

# ============================================================================
# 生成函数
# ============================================================================

def generate_vanilla_ddpm_samples(model, n_samples_per_class, device=DEVICE):
    """使用Vanilla DDPM生成样本"""
    if model is None:
        return None, None
    
    print(f"  生成Vanilla DDPM样本...")
    n_timesteps = 1000
    batch_size = 16
    
    # 计算噪声调度
    steps = np.arange(0, n_timesteps + 1, dtype=np.float32)
    alphas_cumprod = np.cos(((steps / n_timesteps) + 0.008) / 1.008 * (np.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0.0001, 0.02)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    betas = torch.from_numpy(betas).float().to(device)
    alphas = torch.from_numpy(alphas).float().to(device)
    alphas_cumprod = torch.from_numpy(alphas_cumprod).float().to(device)
    
    generated_X = []
    generated_y = []
    
    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            class_samples = []
            n_batches = (n_samples_per_class + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                current_batch_size = min(batch_size, n_samples_per_class - len(class_samples))
                x = torch.randn(current_batch_size, C, T, device=device)
                y = torch.full((current_batch_size,), class_id, device=device, dtype=torch.long)
                
                for t_idx in reversed(range(n_timesteps)):
                    t = torch.full((current_batch_size,), t_idx, device=device, dtype=torch.long)
                    eps_pred = model(x, t, y)
                    
                    alpha_t = alphas[t_idx]
                    alpha_cumprod_t = alphas_cumprod[t_idx]
                    beta_t = betas[t_idx]
                    
                    if t_idx > 0:
                        alpha_cumprod_t_prev = alphas_cumprod[t_idx - 1]
                    else:
                        alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
                    
                    sqrt_alpha_t = torch.sqrt(alpha_t)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                    
                    pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * eps_pred) / torch.sqrt(alpha_cumprod_t)
                    pred_x0 = torch.clamp(pred_x0, -3, 3)
                    
                    dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * eps_pred
                    x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
                    
                    if t_idx > 0:
                        noise = torch.randn_like(x)
                        sigma_t = torch.sqrt(beta_t)
                        x = x + sigma_t * noise
                
                class_samples.append(x.cpu().numpy())
            
            class_samples = np.concatenate(class_samples, axis=0)[:n_samples_per_class]
            generated_X.append(class_samples)
            generated_y.append(np.full(n_samples_per_class, class_id))
    
    generated_X = np.concatenate(generated_X, axis=0)
    generated_y = np.concatenate(generated_y, axis=0)
    return generated_X, generated_y

def generate_gan_samples(model, n_samples_per_class, device=DEVICE):
    """使用GAN生成样本（与三种场景测试相同的生成方式）"""
    if model is None:
        return None, None
    
    print(f"  生成GAN样本...")
    generated_X = []
    generated_y = []
    
    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            # 使用与评估脚本相同的z_dim=128和批处理方式
            n_batches = (n_samples_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_samples_per_class - len([y for y in generated_y if y == class_id]))
                if batch_size <= 0:
                    break
                z = torch.randn(batch_size, 128, device=device)
                y = torch.full((batch_size,), class_id, device=device, dtype=torch.long)
                x_gen = model(z, y)
                generated_X.append(x_gen.cpu().numpy())
                generated_y.extend([class_id] * batch_size)
    
    generated_X = np.concatenate(generated_X, axis=0)
    generated_y = np.array(generated_y)
    return generated_X, generated_y

def generate_vae_samples(model, n_samples_per_class, device=DEVICE):
    """使用VAE生成样本（与三种场景测试相同的生成方式）"""
    if model is None:
        return None, None
    
    print(f"  生成VAE样本...")
    generated_X = []
    generated_y = []
    
    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            # 使用与评估脚本相同的latent_dim=128和批处理方式
            n_batches = (n_samples_per_class + 49) // 50
            for _ in range(n_batches):
                batch_size = min(50, n_samples_per_class - len([y for y in generated_y if y == class_id]))
                if batch_size <= 0:
                    break
                z = torch.randn(batch_size, 128, device=device)
                y = torch.full((batch_size,), class_id, device=device, dtype=torch.long)
                x_gen = model.decode(z, y)
                generated_X.append(x_gen.cpu().numpy())
                generated_y.extend([class_id] * batch_size)
    
    generated_X = np.concatenate(generated_X, axis=0)
    generated_y = np.array(generated_y)
    return generated_X, generated_y

def generate_gaussian_noise_samples(X_train, y_train, n_samples_per_class, noise_level=0.05):
    """使用高斯噪声生成样本"""
    print(f"  生成高斯噪声样本...")
    generated_X = []
    generated_y = []
    
    for class_id in range(NUM_CLASSES):
        class_data = X_train[y_train == class_id]
        if len(class_data) == 0:
            continue
        
        # 计算该类数据的标准差
        class_std = class_data.std(axis=0, keepdims=True)
        
        for _ in range(n_samples_per_class):
            # 随机选择一个基础样本
            base = class_data[np.random.randint(len(class_data))]
            
            # 添加高斯噪声
            noise = np.random.randn(C, T) * class_std * noise_level
            sample = base + noise
            
            generated_X.append(sample)
            generated_y.append(class_id)
    
    generated_X = np.array(generated_X)
    generated_y = np.array(generated_y)
    return generated_X, generated_y

def generate_smote_samples(X_train, y_train, n_samples_per_class):
    """使用SMOTE生成样本"""
    print(f"  生成SMOTE样本...")
    generated_X = []
    generated_y = []
    
    for class_id in range(NUM_CLASSES):
        class_data = X_train[y_train == class_id]
        if len(class_data) == 0:
            continue
        
        for _ in range(n_samples_per_class):
            # 随机选择一个基础样本
            idx = np.random.randint(len(class_data))
            sample = class_data[idx]
            
            # 找到k个最近邻（简化版：随机选择一个邻居）
            k = min(5, len(class_data) - 1)
            if k > 0:
                neighbor_indices = np.random.choice(
                    [i for i in range(len(class_data)) if i != idx], 
                    k, replace=False
                )
                neighbor = class_data[neighbor_indices[0]]
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
            else:
                synthetic = sample
            
            generated_X.append(synthetic)
            generated_y.append(class_id)
    
    generated_X = np.array(generated_X)
    generated_y = np.array(generated_y)
    return generated_X, generated_y

def normalize_to_match_real(gen_data, real_data):
    """
    将生成数据的统计特性对齐到真实数据
    
    Args:
        gen_data: 生成数据 (n_samples, n_channels, n_timesteps)
        real_data: 真实数据 (n_samples, n_channels, n_timesteps)
    
    Returns:
        normalized_gen_data: 归一化后的生成数据
    """
    # 计算真实数据的全局统计量（按通道）
    real_mean = real_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
    real_std = real_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
    
    # 计算生成数据的全局统计量（按通道）
    gen_mean = gen_data.mean(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
    gen_std = gen_data.std(axis=(0, 2), keepdims=True)       # (1, n_channels, 1)
    
    # 标准化生成数据并重新缩放到真实数据的统计特性
    gen_normalized = (gen_data - gen_mean) / (gen_std + 1e-8)
    gen_matched = gen_normalized * real_std + real_mean
    
    return gen_matched

# ============================================================================
# 评估函数
# ============================================================================

def evaluate_model(ddpm, X_train, y_train, n_samples=100, guidance_scale=2.0, ddim_steps=100, use_ddim=False, match_statistics=False, match_strength=0.3):
    """
    评估模型 - 针对第一个被试第一个会话的数据
    
    Args:
        ddpm: 训练好的DDPM模型
        X_train: 训练数据（第一个被试第一个会话）
        y_train: 训练标签
        n_samples: 每类生成样本数
        guidance_scale: 引导强度
        ddim_steps: DDIM采样步数
        
    Returns:
        results: 评估结果字典
    """
    print("\n" + "=" * 70)
    print("质量评估对比 - 第一个被试第一个会话")
    print("=" * 70)
    
    ddpm.eval()
    metrics = EvaluationMetrics(fs=FS, c3_idx=C3_IDX, c4_idx=C4_IDX)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    # 使用全部真实数据（第一个被试第一个会话的所有288个样本）
    real_data = X_train
    real_labels = y_train
    
    print(f"\n真实数据: {len(real_data)} 个样本（全部使用）")
    print(f"  类别分布: {np.bincount(real_labels)}")
    
    # 生成数据
    print(f"\n生成数据 (guidance_scale={guidance_scale})...")
    gen_data = []
    gen_labels = []
    
    n_per_class = n_samples // NUM_CLASSES
    for cls in range(NUM_CLASSES):
        print(f"  生成类别 {cls} ({class_names[cls]}): {n_per_class} 个样本")
        y_gen = torch.full((n_per_class,), cls, device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            # 根据配置选择采样方法
            if use_ddim:
                # DDIM快速采样（可能偏离训练分布）
                data = ddpm.sample_ddim(n_per_class, y_gen, steps=ddim_steps, guidance_scale=guidance_scale, device=DEVICE)
            else:
                # 标准采样（更接近训练时的分布）
                data = ddpm.sample(n_per_class, y_gen, guidance_scale=guidance_scale, device=DEVICE)
        gen_data.append(data.cpu().numpy())
        gen_labels.extend([cls] * n_per_class)
    
    gen_data = np.concatenate(gen_data)
    gen_labels = np.array(gen_labels)
    
    # 统计特性匹配（改善分布一致性）
    if match_statistics:
        print(f"\n  应用统计特性匹配以改善分布一致性 (强度={match_strength})...")
        
        # 保存原始生成数据
        gen_data_original = gen_data.copy()
        
        # 策略1: 全局统计特性匹配（所有真实数据的统计特性）
        print("    步骤1: 全局统计特性匹配...")
        real_global_mean = real_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
        real_global_std = real_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
        gen_global_mean = gen_data.mean(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
        gen_global_std = gen_data.std(axis=(0, 2), keepdims=True)       # (1, n_channels, 1)
        
        # 全局标准化和重新缩放（应用匹配强度）
        gen_data_normalized = (gen_data - gen_global_mean) / (gen_global_std + 1e-8)
        gen_data_matched = gen_data_normalized * real_global_std + real_global_mean
        
        # 混合原始和匹配后的数据（根据匹配强度）
        gen_data = (1 - match_strength) * gen_data_original + match_strength * gen_data_matched
        
        # 策略2: 按类别微调（保持类别内的统计特性，但使用更小的权重）
        if match_strength > 0.5:  # 只在匹配强度较高时进行类别微调
            print("    步骤2: 按类别微调...")
            for cls in range(NUM_CLASSES):
                mask_real = real_labels == cls
                mask_gen = gen_labels == cls
                if np.sum(mask_real) > 0 and np.sum(mask_gen) > 0:
                    real_cls_data = real_data[mask_real]
                    gen_cls_data = gen_data[mask_gen]
                    
                    # 计算类别内的统计特性
                    real_cls_mean = real_cls_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
                    real_cls_std = real_cls_data.std(axis=(0, 2), keepdims=True)   # (1, n_channels, 1)
                    gen_cls_mean = gen_cls_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
                    gen_cls_std = gen_cls_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
                    
                    # 混合策略：90%全局匹配 + 10%类别匹配
                    alpha = 0.1  # 类别匹配的权重（降低）
                    target_mean = (1 - alpha) * real_global_mean + alpha * real_cls_mean
                    target_std = (1 - alpha) * real_global_std + alpha * real_cls_std
                    
                    # 微调生成数据（应用较小的调整）
                    gen_cls_data_normalized = (gen_cls_data - gen_cls_mean) / (gen_cls_std + 1e-8)
                    gen_cls_data_matched = gen_cls_data_normalized * target_std + target_mean
                    
                    # 只应用轻微的调整
                    gen_data[mask_gen] = 0.9 * gen_data[mask_gen] + 0.1 * gen_cls_data_matched
        
        print("  ✓ 统计特性匹配完成")
    
    # 完整评估
    print("\n" + "-" * 70)
    print("评估指标计算中...")
    print("-" * 70)
    results = metrics.evaluate(real_data, real_labels, gen_data, gen_labels)
    
    # 使用预训练分类器评估分类准确率
    print("\n使用预训练EEGClassifier评估分类准确率...")
    classifier = ddpm.classifier
    classifier.eval()
    
    # 评估真实数据
    with torch.no_grad():
        real_tensor = torch.FloatTensor(real_data).to(DEVICE)
        real_pred = classifier(real_tensor).argmax(dim=1).cpu().numpy()
        real_acc = np.mean(real_pred == real_labels)
        
        # 评估生成数据
        gen_tensor = torch.FloatTensor(gen_data).to(DEVICE)
        gen_pred = classifier(gen_tensor).argmax(dim=1).cpu().numpy()
        gen_acc = np.mean(gen_pred == gen_labels)
    
    # 添加到结果中
    results['eegnet_accuracy'] = {
        'real': real_acc,
        'generated': gen_acc
    }
    
    # ========== 质量对比报告 ==========
    print("\n" + "=" * 70)
    print("质量评估对比报告")
    print("=" * 70)
    
    # 1. 基本统计对比
    print("\n【1. 基本统计对比】")
    print(f"  真实数据: mean={real_data.mean():.6f}, std={real_data.std():.6f}")
    print(f"  生成数据: mean={gen_data.mean():.6f}, std={gen_data.std():.6f}")
    mean_diff = abs(real_data.mean() - gen_data.mean())
    std_diff = abs(real_data.std() - gen_data.std())
    print(f"  均值差异: {mean_diff:.6f}")
    print(f"  标准差差异: {std_diff:.6f}")
    
    # 分布一致性评估
    mean_ratio = mean_diff / (abs(real_data.mean()) + 1e-8)
    std_ratio = std_diff / (real_data.std() + 1e-8)
    print(f"  均值相对差异: {mean_ratio*100:.2f}%")
    print(f"  标准差相对差异: {std_ratio*100:.2f}%")
    if mean_ratio > 0.1 or std_ratio > 0.2:
        print("  [警告] 生成数据与真实数据的统计特性差异较大，可能影响分布一致性")
    else:
        print("  ✓ 生成数据与真实数据的统计特性较为一致")
    
    # 2. ERD侧化指数对比
    print("\n【2. ERD侧化指数对比】")
    target_lat = ddpm.target_laterality.cpu().numpy()
    print(f"{'类别':<15} {'目标值':>10} {'真实值':>10} {'生成值':>10} {'差异':>10}")
    print("-" * 60)
    for cls in range(NUM_CLASSES):
        real_lat = results['per_class_laterality'][cls]['real']
        gen_lat = results['per_class_laterality'][cls]['generated']
        diff = abs(gen_lat - real_lat)
        print(f"{class_names[cls]:<15} {target_lat[cls]:>10.4f} {real_lat:>10.4f} {gen_lat:>10.4f} {diff:>10.4f}")
    
    # 3. 频段功率对比
    print("\n【3. 频段功率对比 (生成/真实)】")
    print(f"{'频段':<15} {'功率比':>10} {'状态':>10}")
    print("-" * 40)
    for band, ratio in results['band_power_ratios'].items():
        status = "✓ 良好" if 0.8 <= ratio <= 1.2 else "✗ 偏差"
        print(f"{band.capitalize():<15} {ratio:>10.2f}x {status:>10}")
    
    # 4. 分类准确率对比（使用预训练EEGClassifier）
    print("\n【4. 分类准确率对比 - 预训练EEGClassifier】")
    print(f"  EEGNet (真实数据):   {results['eegnet_accuracy']['real']*100:>6.2f}%")
    print(f"  EEGNet (生成数据):   {results['eegnet_accuracy']['generated']*100:>6.2f}%")
    print(f"  准确率差异:           {abs(results['eegnet_accuracy']['real'] - results['eegnet_accuracy']['generated'])*100:>6.2f}%")
    
    # 5. LDA分类准确率对比（作为参考）
    print("\n【5. LDA分类准确率对比（参考）】")
    print(f"  LDA (真实数据):      {results['lda_accuracy']['real']*100:>6.2f}%")
    print(f"  LDA (生成数据):      {results['lda_accuracy']['generated']*100:>6.2f}%")
    print(f"  交叉验证 (真实→生成): {results['cross_val_accuracy']*100:>6.2f}%")
    
    # 5. PSD相关性（如果有）
    if 'psd_correlation' in results:
        print("\n【5. 功率谱密度相关性】")
        print(f"  PSD相关性: {results['psd_correlation']:.4f}")
    
    # 6. 频率相似度（如果有）
    if 'frequency_similarity' in results:
        print("\n【6. 频率相似度】")
        print(f"  频率相似度: {results['frequency_similarity']:.4f}")
    
    print("\n" + "=" * 70)
    
    return results, real_data, real_labels, gen_data, gen_labels

# ============================================================================
# 可视化函数
# ============================================================================

def visualize_class_discriminative_ddpm_only(real_data, real_labels, gen_data, gen_labels,
                                             save_path='outputs/figures/class_discriminative_ddpm_only.png',
                                             ddpm=None):
    """可视化Class-Discriminative DDPM的单独结果 - 2x2布局"""
    print("\n生成Class-Discriminative DDPM可视化...")
    
    metrics = EvaluationMetrics(fs=FS)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors = ['red', 'blue', 'green', 'purple']
    
    # 提取分类器特征
    print("  提取分类器中间层特征...")
    real_feat_classifier = None
    gen_feat_classifier = None
    
    if ddpm is not None:
        classifier = ddpm.classifier
        classifier.eval()
        
        with torch.no_grad():
            real_tensor = torch.FloatTensor(real_data).to(DEVICE)
            real_feat_classifier = classifier.extract_features(real_tensor).cpu().numpy()
            
            gen_tensor = torch.FloatTensor(gen_data).to(DEVICE)
            gen_feat_classifier = classifier.extract_features(gen_tensor).cpu().numpy()
    
    # 提取PCA特征
    print("  提取PCA特征...")
    real_data_flat = real_data.reshape(len(real_data), -1)
    gen_data_flat = gen_data.reshape(len(gen_data), -1)
    
    all_data_flat = np.vstack([real_data_flat, gen_data_flat])
    pca = PCA(n_components=min(50, all_data_flat.shape[1], all_data_flat.shape[0] - 1))
    all_data_pca = pca.fit_transform(all_data_flat)
    
    n_real = len(real_data)
    real_feat_pca = all_data_pca[:n_real]
    gen_feat_pca = all_data_pca[n_real:]
    
    # 计算t-SNE（分类器特征）
    if real_feat_classifier is not None and gen_feat_classifier is not None:
        print("  计算t-SNE (分类器特征)...")
        all_feat_classifier = np.vstack([real_feat_classifier, gen_feat_classifier])
        tsne_classifier = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded_classifier = tsne_classifier.fit_transform(all_feat_classifier)
        
        real_emb_classifier = embedded_classifier[:n_real]
        gen_emb_classifier = embedded_classifier[n_real:]
    else:
        real_emb_classifier = None
        gen_emb_classifier = None
    
    # 计算t-SNE（PCA特征）
    print("  计算t-SNE (PCA特征)...")
    all_feat_pca = np.vstack([real_feat_pca, gen_feat_pca])
    tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded_pca = tsne_pca.fit_transform(all_feat_pca)
    
    real_emb_pca = embedded_pca[:n_real]
    gen_emb_pca = embedded_pca[n_real:]
    
    # 绘图：2x2布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. 分类器特征 - 按类别
    if real_emb_classifier is not None and gen_emb_classifier is not None:
        ax = axes[0, 0]
        for cls in range(NUM_CLASSES):
            mask_real = real_labels == cls
            mask_gen = gen_labels == cls
            ax.scatter(real_emb_classifier[mask_real, 0], real_emb_classifier[mask_real, 1],
                       c=colors[cls], marker='o', alpha=0.6, s=40, label=f'{class_names[cls]} (Real)')
            ax.scatter(gen_emb_classifier[mask_gen, 0], gen_emb_classifier[mask_gen, 1],
                       c=colors[cls], marker='x', alpha=0.6, s=40, linewidths=1.5)
        ax.set_title('t-SNE by Class (Classifier Features)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
        ax.grid(True, alpha=0.3)
    
    # 2. 分类器特征 - 真实vs生成
    if real_emb_classifier is not None and gen_emb_classifier is not None:
        ax = axes[0, 1]
        ax.scatter(real_emb_classifier[:, 0], real_emb_classifier[:, 1], 
                   c='blue', marker='o', alpha=0.6, label='Real', s=50)
        ax.scatter(gen_emb_classifier[:, 0], gen_emb_classifier[:, 1], 
                   c='red', marker='x', alpha=0.6, label='Generated', s=50, linewidths=1.5)
        ax.set_title('t-SNE: Real vs Generated (Classifier Features)', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
        ax.grid(True, alpha=0.3)
    
    # 3. PCA特征 - 按类别
    ax = axes[1, 0]
    for cls in range(NUM_CLASSES):
        mask_real = real_labels == cls
        mask_gen = gen_labels == cls
        ax.scatter(real_emb_pca[mask_real, 0], real_emb_pca[mask_real, 1],
                   c=colors[cls], marker='o', alpha=0.6, s=40, label=f'{class_names[cls]} (Real)')
        ax.scatter(gen_emb_pca[mask_gen, 0], gen_emb_pca[mask_gen, 1],
                   c=colors[cls], marker='x', alpha=0.6, s=40, linewidths=1.5)
    ax.set_title('t-SNE by Class (PCA Features)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
    ax.grid(True, alpha=0.3)
    
    # 4. PCA特征 - 真实vs生成
    ax = axes[1, 1]
    ax.scatter(real_emb_pca[:, 0], real_emb_pca[:, 1], 
               c='blue', marker='o', alpha=0.6, label='Real', s=50)
    ax.scatter(gen_emb_pca[:, 0], gen_emb_pca[:, 1], 
               c='red', marker='x', alpha=0.6, label='Generated', s=50, linewidths=1.5)
    ax.set_title('t-SNE: Real vs Generated (PCA Features)', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存PNG
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存PNG: {save_path}")
    
    # 保存PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"保存PDF: {pdf_path}")
    
    plt.close()

def visualize_all_methods_comparison(real_data, real_labels, gen_data_dict, gen_labels_dict,
                                     save_path='outputs/figures/all_methods_comparison.png',
                                     ddpm=None):
    """可视化所有模型的对比 - 每个模型一行，4列"""
    print("\n生成所有模型对比可视化...")
    
    metrics = EvaluationMetrics(fs=FS)
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors = ['red', 'blue', 'green', 'purple']
    method_colors = {'Class-Discriminative DDPM': 'red', 'Vanilla DDPM': 'orange', 
                     'GAN': 'green', 'VAE': 'purple', 'Gaussian Noise': 'brown',
                     'SMOTE': 'pink'}
    method_markers = {'Class-Discriminative DDPM': 'x', 'Vanilla DDPM': '^', 
                      'GAN': 's', 'VAE': 'D', 'Gaussian Noise': 'v', 'SMOTE': '*'}
    
    # 提取分类器特征（仅用于Class-Discriminative DDPM）
    print("  提取分类器中间层特征...")
    real_feat_classifier = None
    gen_feat_classifier_dict = {}
    
    if ddpm is not None:
        classifier = ddpm.classifier
        classifier.eval()
        
        with torch.no_grad():
            real_tensor = torch.FloatTensor(real_data).to(DEVICE)
            real_feat_classifier = classifier.extract_features(real_tensor).cpu().numpy()
            
            # 为每个模型提取分类器特征
            for method_name, gen_data in gen_data_dict.items():
                if gen_data is not None:
                    gen_tensor = torch.FloatTensor(gen_data).to(DEVICE)
                    gen_feat_classifier_dict[method_name] = classifier.extract_features(gen_tensor).cpu().numpy()
    
    # 提取PCA特征（用于所有模型）
    print("  提取PCA特征...")
    real_data_flat = real_data.reshape(len(real_data), -1)
    
    # 收集所有生成数据用于PCA拟合
    all_gen_data_flat = []
    for method_name, gen_data in gen_data_dict.items():
        if gen_data is not None:
            all_gen_data_flat.append(gen_data.reshape(len(gen_data), -1))
    
    if all_gen_data_flat:
        all_data_flat = np.vstack([real_data_flat] + all_gen_data_flat)
        pca = PCA(n_components=min(50, all_data_flat.shape[1], all_data_flat.shape[0] - 1))
        all_data_pca = pca.fit_transform(all_data_flat)
        
        n_real = len(real_data)
        real_feat_pca = all_data_pca[:n_real]
        
        # 为每个模型提取PCA特征
        gen_feat_pca_dict = {}
        start_idx = n_real
        for method_name, gen_data in gen_data_dict.items():
            if gen_data is not None:
                n_gen = len(gen_data)
                gen_feat_pca_dict[method_name] = all_data_pca[start_idx:start_idx+n_gen]
                start_idx += n_gen
    else:
        real_feat_pca = None
        gen_feat_pca_dict = {}
    
    # 计算t-SNE（分类器特征）
    if real_feat_classifier is not None and gen_feat_classifier_dict:
        print("  计算t-SNE (分类器特征)...")
        all_feat_classifier = [real_feat_classifier]
        for method_name, gen_feat in gen_feat_classifier_dict.items():
            if gen_feat is not None:
                all_feat_classifier.append(gen_feat)
        
        all_feat_classifier = np.vstack(all_feat_classifier)
        tsne_classifier = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded_classifier = tsne_classifier.fit_transform(all_feat_classifier)
        
        n_real = len(real_data)
        real_emb_classifier = embedded_classifier[:n_real]
        
        gen_emb_classifier_dict = {}
        start_idx = n_real
        for method_name, gen_data in gen_data_dict.items():
            if gen_data is not None and method_name in gen_feat_classifier_dict:
                n_gen = len(gen_data)
                gen_emb_classifier_dict[method_name] = embedded_classifier[start_idx:start_idx+n_gen]
                start_idx += n_gen
    else:
        real_emb_classifier = None
        gen_emb_classifier_dict = {}
    
    # 计算t-SNE（PCA特征）
    if real_feat_pca is not None and gen_feat_pca_dict:
        print("  计算t-SNE (PCA特征)...")
        all_feat_pca = [real_feat_pca]
        for method_name, gen_feat in gen_feat_pca_dict.items():
            if gen_feat is not None:
                all_feat_pca.append(gen_feat)
        
        all_feat_pca = np.vstack(all_feat_pca)
        tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded_pca = tsne_pca.fit_transform(all_feat_pca)
        
        n_real = len(real_data)
        real_emb_pca = embedded_pca[:n_real]
        
        gen_emb_pca_dict = {}
        start_idx = n_real
        for method_name, gen_data in gen_data_dict.items():
            if gen_data is not None and method_name in gen_feat_pca_dict:
                n_gen = len(gen_data)
                gen_emb_pca_dict[method_name] = embedded_pca[start_idx:start_idx+n_gen]
                start_idx += n_gen
    else:
        real_emb_pca = None
        gen_emb_pca_dict = {}
    
    # 计算全局坐标范围（用于标准化所有子图）
    print("  计算全局坐标范围以标准化所有子图...")
    
    # 分类器特征的全局范围
    classifier_x_min, classifier_x_max = None, None
    classifier_y_min, classifier_y_max = None, None
    if real_emb_classifier is not None:
        classifier_x_min = real_emb_classifier[:, 0].min()
        classifier_x_max = real_emb_classifier[:, 0].max()
        classifier_y_min = real_emb_classifier[:, 1].min()
        classifier_y_max = real_emb_classifier[:, 1].max()
    
    for method_name, gen_emb in gen_emb_classifier_dict.items():
        if gen_emb is not None:
            if classifier_x_min is None:
                classifier_x_min = gen_emb[:, 0].min()
                classifier_x_max = gen_emb[:, 0].max()
                classifier_y_min = gen_emb[:, 1].min()
                classifier_y_max = gen_emb[:, 1].max()
            else:
                classifier_x_min = min(classifier_x_min, gen_emb[:, 0].min())
                classifier_x_max = max(classifier_x_max, gen_emb[:, 0].max())
                classifier_y_min = min(classifier_y_min, gen_emb[:, 1].min())
                classifier_y_max = max(classifier_y_max, gen_emb[:, 1].max())
    
    # PCA特征的全局范围
    pca_x_min, pca_x_max = None, None
    pca_y_min, pca_y_max = None, None
    if real_emb_pca is not None:
        pca_x_min = real_emb_pca[:, 0].min()
        pca_x_max = real_emb_pca[:, 0].max()
        pca_y_min = real_emb_pca[:, 1].min()
        pca_y_max = real_emb_pca[:, 1].max()
    
    for method_name, gen_emb in gen_emb_pca_dict.items():
        if gen_emb is not None:
            if pca_x_min is None:
                pca_x_min = gen_emb[:, 0].min()
                pca_x_max = gen_emb[:, 0].max()
                pca_y_min = gen_emb[:, 1].min()
                pca_y_max = gen_emb[:, 1].max()
            else:
                pca_x_min = min(pca_x_min, gen_emb[:, 0].min())
                pca_x_max = max(pca_x_max, gen_emb[:, 0].max())
                pca_y_min = min(pca_y_min, gen_emb[:, 1].min())
                pca_y_max = max(pca_y_max, gen_emb[:, 1].max())
    
    # 添加一些边距（5%）
    if classifier_x_min is not None:
        classifier_x_range = classifier_x_max - classifier_x_min
        classifier_y_range = classifier_y_max - classifier_y_min
        classifier_x_min -= classifier_x_range * 0.05
        classifier_x_max += classifier_x_range * 0.05
        classifier_y_min -= classifier_y_range * 0.05
        classifier_y_max += classifier_y_range * 0.05
    
    if pca_x_min is not None:
        pca_x_range = pca_x_max - pca_x_min
        pca_y_range = pca_y_max - pca_y_min
        pca_x_min -= pca_x_range * 0.05
        pca_x_max += pca_x_range * 0.05
        pca_y_min -= pca_y_range * 0.05
        pca_y_max += pca_y_range * 0.05
    
    # 绘图：为每个模型创建单独的子图组
    # 每个模型一行，每行4个子图（分类器特征按类别、分类器特征真实vs生成、PCA特征按类别、PCA特征真实vs生成）
    n_methods = len([m for m in gen_data_dict.keys() if gen_data_dict[m] is not None])
    
    if n_methods == 0:
        print("[警告] 没有可用的生成数据，跳过可视化")
        return
    
    # 创建子图：n_methods行 x 4列
    fig, axes = plt.subplots(n_methods, 4, figsize=(20, 5 * n_methods))
    
    # 确保axes是二维数组
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    method_idx = 0
    for method_name in gen_data_dict.keys():
        if gen_data_dict[method_name] is None:
            continue
        
        # 获取该模型的嵌入数据
        gen_emb_classifier = gen_emb_classifier_dict.get(method_name)
        gen_emb_pca = gen_emb_pca_dict.get(method_name)
        gen_labels = gen_labels_dict.get(method_name)
        
        # 第1列：分类器特征 - 按类别
        if real_emb_classifier is not None and gen_emb_classifier is not None:
            ax = axes[method_idx, 0]
            # 先画真实数据
            for cls in range(NUM_CLASSES):
                mask_real = real_labels == cls
                ax.scatter(real_emb_classifier[mask_real, 0], real_emb_classifier[mask_real, 1],
                           c=colors[cls], marker='o', alpha=0.6, s=40, label=f'{class_names[cls]} (Real)')
            
            # 再画该模型的生成数据
            if gen_labels is not None:
                for cls in range(NUM_CLASSES):
                    mask_gen = gen_labels == cls
                    if np.sum(mask_gen) > 0:
                        ax.scatter(gen_emb_classifier[mask_gen, 0], gen_emb_classifier[mask_gen, 1],
                                   c=colors[cls], marker='x', alpha=0.6, s=40, linewidths=1.5,
                                   label=f'{class_names[cls]} (Gen)')
            
            ax.set_title(f'{method_name}\nClassifier Features by Class', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            # 图例放在子图内部，使用两列排列，放在右下角避免遮挡数据
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
            ax.grid(True, alpha=0.3)
            # 设置统一的坐标轴范围
            if classifier_x_min is not None:
                ax.set_xlim(classifier_x_min, classifier_x_max)
                ax.set_ylim(classifier_y_min, classifier_y_max)
        
        # 第2列：分类器特征 - 真实vs生成
        if real_emb_classifier is not None and gen_emb_classifier is not None:
            ax = axes[method_idx, 1]
            ax.scatter(real_emb_classifier[:, 0], real_emb_classifier[:, 1], 
                       c='blue', marker='o', alpha=0.6, label='Real', s=50)
            ax.scatter(gen_emb_classifier[:, 0], gen_emb_classifier[:, 1], 
                       c=method_colors.get(method_name, 'red'), marker='x', 
                       alpha=0.6, label='Generated', s=50, linewidths=1.5)
            
            ax.set_title(f'{method_name}\nClassifier Features: Real vs Generated', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            # 图例放在子图内部，使用两列排列，放在右下角避免遮挡数据
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
            ax.grid(True, alpha=0.3)
            # 设置统一的坐标轴范围
            if classifier_x_min is not None:
                ax.set_xlim(classifier_x_min, classifier_x_max)
                ax.set_ylim(classifier_y_min, classifier_y_max)
        
        # 第3列：PCA特征 - 按类别
        if real_emb_pca is not None and gen_emb_pca is not None:
            ax = axes[method_idx, 2]
            # 先画真实数据
            for cls in range(NUM_CLASSES):
                mask_real = real_labels == cls
                ax.scatter(real_emb_pca[mask_real, 0], real_emb_pca[mask_real, 1],
                           c=colors[cls], marker='o', alpha=0.6, s=40, label=f'{class_names[cls]} (Real)')
            
            # 再画该模型的生成数据
            if gen_labels is not None:
                for cls in range(NUM_CLASSES):
                    mask_gen = gen_labels == cls
                    if np.sum(mask_gen) > 0:
                        ax.scatter(gen_emb_pca[mask_gen, 0], gen_emb_pca[mask_gen, 1],
                                   c=colors[cls], marker='x', alpha=0.6, s=40, linewidths=1.5,
                                   label=f'{class_names[cls]} (Gen)')
            
            ax.set_title(f'{method_name}\nPCA Features by Class', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            # 图例放在子图内部，使用两列排列，放在右下角避免遮挡数据
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
            ax.grid(True, alpha=0.3)
            # 设置统一的坐标轴范围
            if pca_x_min is not None:
                ax.set_xlim(pca_x_min, pca_x_max)
                ax.set_ylim(pca_y_min, pca_y_max)
        
        # 第4列：PCA特征 - 真实vs生成
        if real_emb_pca is not None and gen_emb_pca is not None:
            ax = axes[method_idx, 3]
            ax.scatter(real_emb_pca[:, 0], real_emb_pca[:, 1], 
                       c='blue', marker='o', alpha=0.6, label='Real', s=50)
            ax.scatter(gen_emb_pca[:, 0], gen_emb_pca[:, 1], 
                       c=method_colors.get(method_name, 'red'), marker='x', 
                       alpha=0.6, label='Generated', s=50, linewidths=1.5)
            
            ax.set_title(f'{method_name}\nPCA Features: Real vs Generated', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            # 图例放在子图内部，使用两列排列，放在右下角避免遮挡数据
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2, columnspacing=0.5, handletextpad=0.3)
            ax.grid(True, alpha=0.3)
            # 设置统一的坐标轴范围
            if pca_x_min is not None:
                ax.set_xlim(pca_x_min, pca_x_max)
                ax.set_ylim(pca_y_min, pca_y_max)
        
        method_idx += 1
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存PNG
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存PNG: {save_path}")
    
    # 保存PDF
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"保存PDF: {pdf_path}")
    
    plt.close()

# ============================================================================
# 缓存函数
# ============================================================================

def get_cache_path(method_name):
    """获取缓存文件路径"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # 将方法名中的空格替换为下划线，避免文件名问题
    method_name_safe = method_name.replace(' ', '_').replace('-', '_')
    # 使用参数作为缓存键的一部分
    cache_key = f"tsne_{method_name_safe}_n{N_SAMPLES}_g{GUIDANCE_SCALE}_ddim{USE_DDIM}_steps{DDIM_STEPS}.npz"
    return os.path.join(CACHE_DIR, cache_key)

def save_generated_data_cache(method_name, gen_data, gen_labels):
    """保存生成数据到缓存"""
    cache_path = get_cache_path(method_name)
    np.savez_compressed(cache_path, data=gen_data, labels=gen_labels)
    print(f"  [缓存] 已保存 {method_name} 数据到: {cache_path}")

def load_generated_data_cache(method_name):
    """从缓存加载生成数据"""
    cache_path = get_cache_path(method_name)
    if os.path.exists(cache_path):
        try:
            cache = np.load(cache_path)
            gen_data = cache['data']
            gen_labels = cache['labels']
            print(f"  [缓存] 从缓存加载 {method_name} 数据: {cache_path}")
            return gen_data, gen_labels
        except Exception as e:
            print(f"  [警告] 加载缓存失败: {e}")
            return None, None
    else:
        print(f"  [缓存] 缓存文件不存在: {cache_path}")
    return None, None

# ============================================================================
# 主函数
# ============================================================================

def main(use_cache=True, force_regenerate=False):
    """主函数 - 评估第一个被试第一个会话的数据生成质量"""
    print("=" * 70)
    print("Class-Discriminative DDPM 质量评估")
    print("=" * 70)
    print("评估目标: 第一个被试 (Subject 0) 第一个会话 (Session 0)")
    print(f"设备: {DEVICE}")
    print(f"生成样本数: {N_SAMPLES} (每类 {N_SAMPLES // NUM_CLASSES} 个)")
    print(f"引导强度: {GUIDANCE_SCALE} (降低以改善分布一致性)")
    print(f"采样方法: {'DDIM快速采样' if USE_DDIM else '标准采样（更接近训练分布）'}")
    if USE_DDIM:
        print(f"DDIM采样步数: {DDIM_STEPS}")
    print(f"统计特性匹配: {'启用' if MATCH_STATISTICS else '禁用'} (改善分布一致性)")
    print(f"特征提取: {'分类器中间层特征' if USE_CLASSIFIER_FEATURES else '频段功率特征'} ({'提升类别判别性' if USE_CLASSIFIER_FEATURES else '保持分布一致性'})")
    print("=" * 70)
    
    # 加载数据（第一个被试第一个会话）
    X_norm, y_train, X_mean, X_std = load_data()
    
    # 加载所有模型
    print("\n" + "-" * 70)
    print("加载已训练的模型...")
    print("-" * 70)
    
    # 加载Class-Discriminative DDPM
    ddpm = load_trained_ddpm(X_norm, y_train, device=DEVICE)
    if ddpm is None:
        print("[错误] 无法加载Class-Discriminative DDPM模型，退出")
        print("   请确保已运行 train_class_discriminative_ddpm.py 训练模型")
        return
    
    # 加载其他模型
    vanilla_ddpm = load_vanilla_ddpm(device=DEVICE)
    gan_model = load_gan_model(device=DEVICE)
    vae_model = load_vae_model(device=DEVICE)
    
    # 评估Class-Discriminative DDPM（带缓存）
    print("\n" + "-" * 70)
    print("评估Class-Discriminative DDPM...")
    print("-" * 70)
    
    # 获取真实数据（总是需要）
    real_data = X_norm
    real_labels = y_train
    
    # 尝试从缓存加载生成数据
    gen_data_cd, gen_labels_cd = None, None
    if use_cache and not force_regenerate:
        gen_data_cd, gen_labels_cd = load_generated_data_cache('Class-Discriminative DDPM')
    
    if gen_data_cd is None:
        print("  [生成] 生成新的数据...")
        results, _, _, gen_data_cd, gen_labels_cd = evaluate_model(
            ddpm, X_norm, y_train,
            n_samples=N_SAMPLES,
            guidance_scale=GUIDANCE_SCALE,
            ddim_steps=DDIM_STEPS,
            use_ddim=USE_DDIM,
            match_statistics=MATCH_STATISTICS,
            match_strength=MATCH_STRENGTH
        )
        # 保存到缓存
        if use_cache:
            save_generated_data_cache('Class-Discriminative DDPM', gen_data_cd, gen_labels_cd)
    else:
        print("  [缓存] 跳过数据生成，直接使用缓存数据")
    
    # 生成其他模型的数据（带缓存）
    print("\n" + "-" * 70)
    print("生成其他模型的数据...")
    print("-" * 70)
    
    n_samples_per_class = N_SAMPLES // NUM_CLASSES
    gen_data_dict = {'Class-Discriminative DDPM': gen_data_cd}
    gen_labels_dict = {'Class-Discriminative DDPM': gen_labels_cd}
    
    # 生成Vanilla DDPM数据
    if vanilla_ddpm is not None:
        gen_data_vd, gen_labels_vd = load_generated_data_cache('Vanilla DDPM') if use_cache and not force_regenerate else (None, None)
        if gen_data_vd is None:
            gen_data_vd, gen_labels_vd = generate_vanilla_ddpm_samples(vanilla_ddpm, n_samples_per_class, device=DEVICE)
            if use_cache and gen_data_vd is not None:
                save_generated_data_cache('Vanilla DDPM', gen_data_vd, gen_labels_vd)
        if gen_data_vd is not None:
            gen_data_dict['Vanilla DDPM'] = gen_data_vd
            gen_labels_dict['Vanilla DDPM'] = gen_labels_vd
    
    # 生成GAN数据
    if gan_model is not None:
        gen_data_gan, gen_labels_gan = load_generated_data_cache('GAN') if use_cache and not force_regenerate else (None, None)
        if gen_data_gan is None:
            gen_data_gan, gen_labels_gan = generate_gan_samples(gan_model, n_samples_per_class, device=DEVICE)
            if use_cache and gen_data_gan is not None:
                save_generated_data_cache('GAN', gen_data_gan, gen_labels_gan)
        if gen_data_gan is not None:
            gen_data_dict['GAN'] = gen_data_gan
            gen_labels_dict['GAN'] = gen_labels_gan
    
    # 生成VAE数据
    if vae_model is not None:
        gen_data_vae, gen_labels_vae = load_generated_data_cache('VAE') if use_cache and not force_regenerate else (None, None)
        if gen_data_vae is None:
            gen_data_vae, gen_labels_vae = generate_vae_samples(vae_model, n_samples_per_class, device=DEVICE)
            if use_cache and gen_data_vae is not None:
                save_generated_data_cache('VAE', gen_data_vae, gen_labels_vae)
        if gen_data_vae is not None:
            gen_data_dict['VAE'] = gen_data_vae
            gen_labels_dict['VAE'] = gen_labels_vae
    
    # 生成传统方法数据（高斯噪声和SMOTE）
    gen_data_gaussian, gen_labels_gaussian = load_generated_data_cache('Gaussian Noise') if use_cache and not force_regenerate else (None, None)
    if gen_data_gaussian is None:
        gen_data_gaussian, gen_labels_gaussian = generate_gaussian_noise_samples(
            X_norm, y_train, n_samples_per_class, noise_level=0.05
        )
        if use_cache:
            save_generated_data_cache('Gaussian Noise', gen_data_gaussian, gen_labels_gaussian)
    gen_data_dict['Gaussian Noise'] = gen_data_gaussian
    gen_labels_dict['Gaussian Noise'] = gen_labels_gaussian
    
    gen_data_smote, gen_labels_smote = load_generated_data_cache('SMOTE') if use_cache and not force_regenerate else (None, None)
    if gen_data_smote is None:
        gen_data_smote, gen_labels_smote = generate_smote_samples(
            X_norm, y_train, n_samples_per_class
        )
        if use_cache:
            save_generated_data_cache('SMOTE', gen_data_smote, gen_labels_smote)
    gen_data_dict['SMOTE'] = gen_data_smote
    gen_labels_dict['SMOTE'] = gen_labels_smote
    
    # 对所有生成数据进行归一化对齐（使其与真实数据的统计特性匹配）
    print("\n" + "-" * 70)
    print("对所有生成数据进行归一化对齐...")
    print("-" * 70)
    for method_name, gen_data in gen_data_dict.items():
        if gen_data is not None:
            print(f"  对齐 {method_name}...")
            gen_data_dict[method_name] = normalize_to_match_real(gen_data, real_data)
            print(f"    对齐前: mean={gen_data.mean():.6f}, std={gen_data.std():.6f}")
            print(f"    对齐后: mean={gen_data_dict[method_name].mean():.6f}, std={gen_data_dict[method_name].std():.6f}")
    
    # 过滤方法：只保留DDPM、GAN、VAE、Gaussian Noise
    print("\n" + "-" * 70)
    print("过滤方法：只保留DDPM、GAN、VAE、Gaussian Noise...")
    print("-" * 70)
    methods_to_keep = ['Class-Discriminative DDPM', 'GAN', 'VAE', 'Gaussian Noise']
    filtered_gen_data_dict = {k: v for k, v in gen_data_dict.items() if k in methods_to_keep}
    filtered_gen_labels_dict = {k: v for k, v in gen_labels_dict.items() if k in methods_to_keep}
    
    print(f"保留的方法: {list(filtered_gen_data_dict.keys())}")
    
    # 可视化 - 所有模型对比
    print("\n" + "-" * 70)
    visualize_all_methods_comparison(
        real_data, real_labels, filtered_gen_data_dict, filtered_gen_labels_dict,
        save_path=f'{OUTPUT_DIR}/all_methods_comparison_subject0_session0.png',
        ddpm=ddpm
    )
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print(f"所有模型对比可视化: {OUTPUT_DIR}/all_methods_comparison_subject0_session0.png")
    print("=" * 70)

if __name__ == '__main__':
    main()

