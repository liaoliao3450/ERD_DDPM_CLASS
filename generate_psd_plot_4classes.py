"""
生成4个类别的PSD对比图
用于论文中的PSD可视化
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 导入数据加载
from utils.data_loader import load_bci2a_data, get_subject_data

# 导入DDPM模型
sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'models' / 'ddpm'))
from class_discriminative import MultiScaleCondUNet, EEGClassifier, ClassDiscriminativeDDPM

def load_real_data(subject_id=1, session_id=1):
    """加载真实实验数据"""
    print(f"加载被试 {subject_id} 会话 {session_id} 的真实数据...")
    
    # 加载所有数据
    X, y, subjects, sessions = load_bci2a_data()
    
    # 获取特定被试和会话的数据
    mask = (subjects == subject_id) & (sessions == session_id)
    X_subj = X[mask]
    y_subj = y[mask]
    
    print(f"  数据形状: {X_subj.shape}, 标签: {np.bincount(y_subj)}")
    return X_subj, y_subj

def load_ddpm_model():
    """加载训练好的DDPM模型"""
    print("加载DDPM模型...")
    checkpoint_path = 'checkpoints/trained_ddpm.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"  警告: DDPM模型不存在: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 重建模型
    eps = MultiScaleCondUNet(channels=22, num_classes=4).to(DEVICE)
    clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
    
    # 从 checkpoint 中读取 target_psd / target_laterality（如果存在）
    if isinstance(checkpoint, dict) and 'target_psd' in checkpoint and 'target_laterality' in checkpoint:
        target_psd = checkpoint['target_psd'].to(DEVICE)
        target_laterality = checkpoint['target_laterality'].to(DEVICE)
    else:
        # 兜底：如果老 checkpoint 里没有这些字段，就用零向量
        target_psd = torch.zeros(501).to(DEVICE)
        target_laterality = torch.zeros(4).to(DEVICE)
    
    ddpm = ClassDiscriminativeDDPM(
        eps, clf,
        target_psd, target_laterality,
        n_timesteps=1000, channels=22, n_samples=1000, fs=250
    ).to(DEVICE)
    
    # 加载 state_dict（优先用 'model_state_dict'，如果失败则用 strict=False）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        try:
            ddpm.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
                print("  警告: checkpoint 参数不完全匹配，使用 strict=False 加载 ...")
                ddpm.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                raise
    else:
        ddpm.load_state_dict(checkpoint, strict=False)
    
    ddpm.eval()
    print("  ✓ DDPM模型加载成功")
    return ddpm

def normalize_generated_data_to_real_stats(real_data, gen_data):
    """
    将生成数据的统计特性对齐到真实数据（逐通道标准化空间）
    
    Args:
        real_data: 真实数据 [N, C, T]
        gen_data: 生成数据 [N, C, T]
    
    Returns:
        gen_data_aligned: 对齐后的生成数据 [N, C, T]
    """
    real_data = real_data.astype(np.float32)
    gen_data = gen_data.astype(np.float32)
    
    # 逐通道统计 (1, C, 1)
    real_mean = real_data.mean(axis=(0, 2), keepdims=True)
    real_std = real_data.std(axis=(0, 2), keepdims=True)
    gen_mean = gen_data.mean(axis=(0, 2), keepdims=True)
    gen_std = gen_data.std(axis=(0, 2), keepdims=True)
    
    eps = 1e-8
    if float(np.max(gen_std)) < eps:
        print("⚠️  生成数据逐通道标准差过小，跳过对齐")
        return gen_data
    
    # 生成数据先做逐通道标准化，再对齐到真实训练集统计量
    X_gen_norm = (gen_data - gen_mean) / (gen_std + eps)
    X_gen_aligned = X_gen_norm * (real_std + eps) + real_mean
    return X_gen_aligned

def generate_ddpm_samples(ddpm, n_samples_per_class=25):
    """使用DDPM生成样本"""
    print(f"  使用DDPM生成 {n_samples_per_class * 4} 个样本（每类 {n_samples_per_class} 个）...")
    ddpm.eval()
    gen_X, gen_y = [], []
    
    with torch.no_grad():
        for c in range(4):
            yg = torch.full((n_samples_per_class,), c, dtype=torch.long, device=DEVICE)
            samples = ddpm.sample_ddim(n_samples_per_class, yg, steps=50, guidance_scale=5.2)
            gen_X.append(samples.cpu().numpy())
            gen_y.extend([c] * n_samples_per_class)
    
    return np.concatenate(gen_X), np.array(gen_y)

def compute_psd(data, fs=250, nperseg=256):
    """计算功率谱密度"""
    # 对所有通道和样本计算平均PSD
    psds = []
    for i in range(len(data)):
        # 对所有通道平均
        channel_psds = []
        for ch in range(data.shape[1]):
            f, psd = signal.welch(data[i, ch, :], fs=fs, nperseg=nperseg)
            channel_psds.append(psd)
        psds.append(np.mean(channel_psds, axis=0))
    
    # 对所有样本平均
    mean_psd = np.mean(psds, axis=0)
    return f, mean_psd

def generate_psd_plot(output_dir, real_data, real_labels, gen_data, gen_labels):
    """生成4个类别的PSD对比图"""
    print("\n[1/2] 生成PSD对比图（4个类别）...")
    
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    colors_real = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝色、橙色、绿色、红色
    colors_gen = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']    # 浅色版本
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Power Spectral Density Comparison: Real vs DDPM-Generated', 
                 fontsize=12, fontweight='bold', y=0.995)
    
    for c in range(4):
        ax = axes[c // 2, c % 2]
        
        # 真实数据PSD
        real_mask = real_labels == c
        if np.sum(real_mask) > 0:
            real_class_data = real_data[real_mask]
            f_real, psd_real = compute_psd(real_class_data, fs=250, nperseg=256)
            # 转换为dB
            psd_real_db = 10 * np.log10(psd_real + 1e-12)
            ax.plot(f_real, psd_real_db, color=colors_real[c], linewidth=2.5, 
                   label='Real', alpha=0.9)
        
        # 生成数据PSD
        gen_mask = gen_labels == c
        if np.sum(gen_mask) > 0:
            gen_class_data = gen_data[gen_mask]
            f_gen, psd_gen = compute_psd(gen_class_data, fs=250, nperseg=256)
            # 转换为dB
            psd_gen_db = 10 * np.log10(psd_gen + 1e-12)
            ax.plot(f_gen, psd_gen_db, color=colors_gen[c], linewidth=2.5, 
                   label='DDPM-Generated', alpha=0.9, linestyle='--')
        
        ax.set_title(f'{class_names[c]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim([0, 40])
        ax.set_ylim([-40, 20])
        
        # 添加频段标注
        ax.axvspan(8, 13, alpha=0.1, color='blue', label='Alpha')
        ax.axvspan(13, 30, alpha=0.1, color='green', label='Beta')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig3_psd_comparison_4classes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ PSD对比图已保存到: {output_path}")
    
    # 同时保存PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"  ✅ PSD对比图PDF已保存到: {pdf_path}")
    
    # 同时保存到paper目录
    paper_fig_path = PROJECT_ROOT / 'paper' / 'fig3_psd_comparison.png'
    plt.savefig(paper_fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已复制到论文目录: {paper_fig_path}")
    
    # 同时保存PDF到paper目录
    paper_pdf_path = PROJECT_ROOT / 'paper' / 'fig3_psd_comparison.pdf'
    plt.savefig(paper_pdf_path, bbox_inches='tight', format='pdf')
    print(f"  ✅ PDF已复制到论文目录: {paper_pdf_path}")
    
    plt.close()

def main():
    print("="*70)
    print("生成4个类别的PSD对比图")
    print("="*70)
    
    output_dir = 'paper_results/figures'
    
    # 加载真实数据
    print("\n[0/2] 加载真实实验数据...")
    real_data, real_labels = load_real_data(subject_id=1, session_id=1)
    
    # 加载DDPM模型并生成样本
    print("\n[0.5/2] 加载DDPM模型并生成样本...")
    ddpm = load_ddpm_model()
    if ddpm is None:
        print("❌ 无法加载DDPM模型，退出")
        return
    
    gen_data, gen_labels = generate_ddpm_samples(ddpm, n_samples_per_class=25)
    
    # 将生成数据对齐到真实数据的统计特性
    print("\n[0.75/2] 对齐生成数据到真实数据的统计特性...")
    gen_data = normalize_generated_data_to_real_stats(real_data, gen_data)
    
    # 生成PSD对比图
    generate_psd_plot(output_dir, real_data, real_labels, gen_data, gen_labels)
    
    print("\n" + "="*70)
    print("✅ 所有图片生成完成！")
    print("="*70)

if __name__ == '__main__':
    main()

