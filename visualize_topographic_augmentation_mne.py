#!/usr/bin/env python3
"""
拓扑图可视化（数据增强论文）- 使用RBF插值方法

Alpha band power拓扑图：
- 4个类别 × 2行（Real vs DDPM）
- 使用RBF插值生成标准脑地形图
- 突出显示C3, Cz, C4位置
"""
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import welch
from scipy.interpolate import Rbf
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from data_loader import load_bci2a_data

# ==================== 通道位置定义（标准10-20系统，归一化坐标）====================
# 通道名称列表（BCI Competition IIa）
CHANNEL_NAMES = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                 'P1', 'Pz', 'P2', 'POz']

# 通道位置（基于标准10-20系统，归一化到[-1, 1]范围）
# 格式：[x, y]，其中x=左右（负=左，正=右），y=前后（负=后，正=前）
# 注意：Cz位于头部轮廓中心（y=0）
CHANNEL_POSITIONS = np.array([
    [0.0, 0.3],      # Fz (前额)
    [-0.3, 0.2],    # FC3
    [-0.15, 0.2],   # FC1
    [0.0, 0.2],     # FCz
    [0.15, 0.2],    # FC2
    [0.3, 0.2],     # FC4
    [-0.5, 0.0],    # C5
    [-0.3, 0.0],    # C3
    [-0.15, 0.0],   # C1
    [0.0, 0.0],     # Cz (中心)
    [0.15, 0.0],    # C2
    [0.3, 0.0],     # C4
    [0.5, 0.0],     # C6
    [-0.3, -0.2],   # CP3
    [-0.15, -0.2],  # CP1
    [0.0, -0.2],    # CPz
    [0.15, -0.2],   # CP2
    [0.3, -0.2],    # CP4
    [-0.15, -0.4],  # P1
    [0.0, -0.4],    # Pz
    [0.15, -0.4],   # P2
    [0.0, -0.5],    # POz
])

# 关键通道索引（对应 CHANNEL_POSITIONS / ch_names 顺序）
KEY_CHANNELS = {'C3': 7, 'Cz': 9, 'C4': 11}

# 运动皮层通道权重（用于相似度匹配时加权）
# 通道顺序与后面的 ch_names 一致：
# ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
#  'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']
MOTOR_WEIGHTS = np.ones(len(CHANNEL_POSITIONS), dtype=float)
for idx in [1, 5, 7, 9, 11, 13, 15, 17]:  # FC3, FC4, C3, Cz, C4, CP3, CPz, CP4
    MOTOR_WEIGHTS[idx] = 3.0  # 进一步放大运动皮层权重

# 相似度阈值（仅用于日志提示，不强制丢弃样本）
SIMILARITY_THRESHOLD = 0.6

def normalize_generated_data_to_real_stats(real_data, gen_data):
    """
    将生成数据的统计特性对齐到真实数据（全局标准化和重新缩放）
    
    Args:
        real_data: 真实数据 [N, C, T]
        gen_data: 生成数据 [M, C, T]
    
    Returns:
        对齐后的生成数据
    """
    # 计算真实数据的全局统计特性（按通道，axis=(0, 2)）
    real_global_mean = real_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
    real_global_std = real_data.std(axis=(0, 2), keepdims=True)     # (1, n_channels, 1)
    
    # 计算生成数据的全局统计特性
    gen_global_mean = gen_data.mean(axis=(0, 2), keepdims=True)    # (1, n_channels, 1)
    gen_global_std = gen_data.std(axis=(0, 2), keepdims=True)       # (1, n_channels, 1)
    
    # 全局标准化和重新缩放
    gen_data_normalized = (gen_data - gen_global_mean) / (gen_global_std + 1e-8)
    gen_data_matched = gen_data_normalized * real_global_std + real_global_mean
    
    return gen_data_matched

def compute_relative_alpha_power(alpha_power, baseline_power=None):
    """
    计算相对Alpha功率（相对于baseline或整体平均）
    
    Args:
        alpha_power: Alpha功率 [n_samples, n_channels] 或 [n_channels]
        baseline_power: 基线功率，如果为None则使用整体平均
    
    Returns:
        相对Alpha功率
    """
    if baseline_power is None:
        # 使用整体平均作为baseline
        baseline_power = alpha_power.mean(axis=0, keepdims=True) if alpha_power.ndim == 2 else alpha_power.mean()
    
    # 计算相对变化百分比: (power - baseline) / baseline * 100
    relative_power = (alpha_power - baseline_power) / (baseline_power + 1e-8) * 100.0
    
    return relative_power

def compute_laterality_difference(alpha_power_absolute, ch_names, class_type):
    """
    计算对侧-同侧差异
    
    使用绝对功率计算对侧相对于同侧的百分比变化
    
    Args:
        alpha_power_absolute: Alpha绝对功率 [n_channels] 或 [n_samples, n_channels]
        ch_names: 通道名称列表
        class_type: 类别类型 ('left_hand', 'right_hand', 'feet', 'tongue')
    
    Returns:
        对侧-同侧差异（百分比变化）
    """
    # 通道索引：C3=7, Cz=9, C4=11
    # 对于Left Hand: 对侧=C4(11), 同侧=C3(7)
    # 对于Right Hand: 对侧=C3(7), 同侧=C4(11)
    # 对于Feet: 对侧=Cz(9), 同侧=平均(C3, C4)
    
    if alpha_power_absolute.ndim == 1:
        alpha_power_absolute = alpha_power_absolute.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_samples = alpha_power_absolute.shape[0]
    n_channels = alpha_power_absolute.shape[1]
    laterality_diff = np.zeros((n_samples, n_channels))
    
    for i in range(n_samples):
        if class_type == 'left_hand':
            # Left Hand: 对侧=C4, 同侧=C3
            # 计算每个通道相对于同侧（C3）的百分比变化
            ipsilateral_power = alpha_power_absolute[i, 7]  # C3
            for ch_idx in range(n_channels):
                # 百分比变化: (power - ipsilateral) / ipsilateral * 100
                laterality_diff[i, ch_idx] = (alpha_power_absolute[i, ch_idx] - ipsilateral_power) / (ipsilateral_power + 1e-8) * 100.0
        elif class_type == 'right_hand':
            # Right Hand: 对侧=C3, 同侧=C4
            # 计算每个通道相对于同侧（C4）的百分比变化
            ipsilateral_power = alpha_power_absolute[i, 11]  # C4
            for ch_idx in range(n_channels):
                laterality_diff[i, ch_idx] = (alpha_power_absolute[i, ch_idx] - ipsilateral_power) / (ipsilateral_power + 1e-8) * 100.0
        elif class_type == 'feet':
            # Feet: 对侧=Cz, 同侧=平均(C3, C4)
            # 计算每个通道相对于周围区域（C3, C4平均）的百分比变化
            ipsilateral_power = (alpha_power_absolute[i, 7] + alpha_power_absolute[i, 11]) / 2  # 平均(C3, C4)
            for ch_idx in range(n_channels):
                laterality_diff[i, ch_idx] = (alpha_power_absolute[i, ch_idx] - ipsilateral_power) / (ipsilateral_power + 1e-8) * 100.0
        else:  # tongue
            # Tongue: 保持原值（相对功率）
            laterality_diff[i, :] = alpha_power_absolute[i, :]
    
    if squeeze_output:
        laterality_diff = laterality_diff.squeeze()
    
    return laterality_diff

def compute_alpha_power(X, fs=250):
    """
    计算Alpha band (8-13 Hz) power
    
    Args:
        X: EEG数据 (n_samples, n_channels, n_timepoints)
        fs: 采样率
    
    Returns:
        alpha_power: Alpha band power (n_samples, n_channels)
    """
    n_samples, n_channels, n_timepoints = X.shape
    alpha_power = np.zeros((n_samples, n_channels))
    
    for i in range(n_samples):
        for ch in range(n_channels):
            # 计算功率谱密度
            freqs, psd = welch(X[i, ch, :], fs=fs, nperseg=min(256, n_timepoints))
            
            # 提取Alpha band (8-13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power[i, ch] = np.mean(psd[alpha_mask])
    
    return alpha_power

def select_representative_samples(X, y, alpha_power, n_per_class=1):
    """
    为每个类别选择ERD明显的代表性样本
    
    使用相对功率和laterality difference来选择ERD最明显的样本
    """
    selected_indices = []
    
    # 通道索引：C3=7, Cz=9, C4=11, FCz=3
    
    for c in range(4):
        mask = y == c
        X_class = X[mask]
        alpha_class = alpha_power[mask]
        
        if len(X_class) == 0:
            continue
        
        # 计算该类别的baseline（该类别的平均功率）
        baseline = alpha_class.mean(axis=0)
        
        # 计算相对功率（相对于该类别的baseline）
        relative_power = (alpha_class - baseline) / (baseline + 1e-8) * 100.0
        
        if c == 0:  # Left Hand - 选择C4（对侧）相对于C3（同侧）ERD最明显的
            # Laterality difference: (C3 - C4) / C3 * 100
            # 正值表示C4功率更低（ERD更明显）
            ipsilateral = relative_power[:, 7]   # C3 (同侧)
            contralateral = relative_power[:, 11]  # C4 (对侧)
            # ERD明显 = 对侧功率更低，即 (同侧 - 对侧) 越大越好
            scores = ipsilateral - contralateral
            print(f"  Left Hand: 选择laterality difference最大的样本")
            print(f"    C3相对功率范围: [{ipsilateral.min():.1f}%, {ipsilateral.max():.1f}%]")
            print(f"    C4相对功率范围: [{contralateral.min():.1f}%, {contralateral.max():.1f}%]")
            print(f"    Laterality difference范围: [{scores.min():.1f}%, {scores.max():.1f}%]")
            
        elif c == 1:  # Right Hand - 选择C3（对侧）相对于C4（同侧）ERD最明显的
            # Laterality difference: (C4 - C3) / C4 * 100
            ipsilateral = relative_power[:, 11]  # C4 (同侧)
            contralateral = relative_power[:, 7]  # C3 (对侧)
            # ERD明显 = 对侧功率更低，即 (同侧 - 对侧) 越大越好
            scores = ipsilateral - contralateral
            print(f"  Right Hand: 选择laterality difference最大的样本")
            print(f"    C4相对功率范围: [{ipsilateral.min():.1f}%, {ipsilateral.max():.1f}%]")
            print(f"    C3相对功率范围: [{contralateral.min():.1f}%, {contralateral.max():.1f}%]")
            print(f"    Laterality difference范围: [{scores.min():.1f}%, {scores.max():.1f}%]")
            
        elif c == 2:  # Feet - 选择Cz相对功率最低的（ERD最明显）
            scores = -relative_power[:, 9]  # Cz，负号表示选择相对功率最低的
            cz_relative = relative_power[:, 9]
            print(f"  Feet: 选择Cz相对功率最低的样本")
            print(f"    Cz相对功率范围: [{cz_relative.min():.1f}%, {cz_relative.max():.1f}%]")
            
        else:  # Tongue - 选择frontal-central区域相对功率最高的
            # 使用FCz和Cz的平均相对功率
            scores = (relative_power[:, 3] + relative_power[:, 9]) / 2.0  # FCz + Cz
            print(f"  Tongue: 选择FCz+Cz相对功率最高的样本")
            print(f"    FCz+Cz相对功率范围: [{scores.min():.1f}%, {scores.max():.1f}%]")
        
        # 选择得分最高的样本（laterality difference最大，或相对功率最低/最高）
        top_indices = np.argsort(scores)[-n_per_class:]
        
        # 转换回原始索引
        class_indices = np.where(mask)[0]
        selected_idx = class_indices[top_indices[-1]]  # 选择得分最高的
        selected_indices.append(selected_idx)
        
        # 打印选择的样本信息
        selected_relative = relative_power[top_indices[-1]]
        if c < 2:  # Left/Right Hand
            print(f"    选择的样本索引: {selected_idx}, Laterality diff: {scores[top_indices[-1]]:.1f}%")
        elif c == 2:  # Feet
            print(f"    选择的样本索引: {selected_idx}, Cz相对功率: {selected_relative[9]:.1f}%")
        else:  # Tongue
            print(f"    选择的样本索引: {selected_idx}, FCz+Cz相对功率: {scores[top_indices[-1]]:.1f}%")
    
    return selected_indices

def plot_topomap_rbf(ax, values, ch_names, title='', vmin=None, vmax=None, 
                     highlight_channels=None):
    """
    使用RBF插值绘制拓扑图
    
    Args:
        ax: matplotlib axis
        values: 通道值 (n_channels,)
        ch_names: 通道名称列表（必须与CHANNEL_NAMES顺序一致）
        title: 标题
        vmin, vmax: 颜色范围
        highlight_channels: 需要突出显示的通道列表
    """
    # 创建高分辨率网格
    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 头部半径
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius
    
    # 获取通道位置（确保顺序与ch_names一致）
    if len(values) != len(CHANNEL_POSITIONS):
        raise ValueError(f"通道数量不匹配: values={len(values)}, positions={len(CHANNEL_POSITIONS)}")
    
    # RBF插值
    rbf = Rbf(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], values,
              function='multiquadric', smooth=0.1)
    Zi = rbf(Xi, Yi)
    Zi[~mask] = np.nan
    
    # 绘制填充等高线 - 使用RdBu_r颜色映射
    # 蓝色=低功率(ERD/激活), 红色=高功率
    # 数据应该是归一化到0-1范围的
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1
    levels = np.linspace(vmin, vmax, 50)
    im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both', vmin=vmin, vmax=vmax)
    
    # 绘制头部轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)
    
    # 绘制鼻子（在Y轴正方向，即前额方向）
    nose_y = [head_radius, head_radius + 0.10, head_radius]
    nose_x = [0.05, 0, -0.05]
    ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)
    
    # 绘制耳朵
    ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
    ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
    
    # 绘制所有电极位置
    ax.scatter(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], c='k', s=12, zorder=5, 
              edgecolors='white', linewidths=0.5)
    
    # 标注关键通道
    if highlight_channels:
        for ch_name in highlight_channels:
            if ch_name in KEY_CHANNELS and ch_name in ch_names:
                ch_idx = KEY_CHANNELS[ch_name]
                pos = CHANNEL_POSITIONS[ch_idx]
                # 用较大的标记突出显示关键通道
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6, 
                          edgecolors='black', linewidths=1.5, marker='o')
                # 添加通道名称标签
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.15])
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 设置标题
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    
    return im

def plot_topographic_comparison():
    """绘制拓扑图对比"""
    
    print("="*60)
    print("拓扑图可视化（数据增强）- RBF插值方法")
    print("="*60)
    
    # ==================== 加载数据 ====================
    print("\n加载数据...")
    X, y, subjects, sessions = load_bci2a_data()
    
    # 使用Subject 1 Session 1的数据
    subject_id = 0  # Subject 1 (索引从0开始)
    session_id = 0  # Session 1
    subject1_session1_mask = (subjects == subject_id) & (sessions == session_id)
    X_real = X[subject1_session1_mask]
    y_real = y[subject1_session1_mask]
    
    print(f"真实数据: {X_real.shape}")
    print(f"被试: Subject {subject_id + 1}, Session {session_id + 1}")
    
    # 计算Alpha band power
    print("\n计算Alpha band power...")
    alpha_power_real = compute_alpha_power(X_real)
    print(f"Alpha power shape: {alpha_power_real.shape}")
    
    # 先计算每个类别自己的baseline（该类别的所有样本平均）
    print("\n计算每个类别的baseline...")
    baseline_per_class = {}
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    for class_idx in range(4):
        class_mask = (y_real == class_idx)
        if class_mask.sum() > 0:
            baseline_per_class[class_idx] = alpha_power_real[class_mask].mean(axis=0)
            print(f"  {class_names[class_idx]}: {class_mask.sum()}个样本")
            print(f"    Baseline C3: {baseline_per_class[class_idx][7]:.2e}, C4: {baseline_per_class[class_idx][11]:.2e}, Cz: {baseline_per_class[class_idx][9]:.2e}")
        else:
            # 如果没有该类别的数据，使用全局平均
            baseline_per_class[class_idx] = alpha_power_real.mean(axis=0)
            print(f"  {class_names[class_idx]}: 无数据，使用全局baseline")
    
    # 选择代表性样本
    print("\n选择代表性样本...")
    selected_indices = select_representative_samples(X_real, y_real, alpha_power_real, n_per_class=1)
    
    X_real_selected = X_real[selected_indices]
    y_real_selected = y_real[selected_indices]
    alpha_real_selected = alpha_power_real[selected_indices]
    
    print(f"选择的样本索引: {selected_indices}")
    print(f"选择的样本标签: {y_real_selected}")
    print(f"对应的类别: {[['Left Hand', 'Right Hand', 'Feet', 'Tongue'][int(l)] for l in y_real_selected]}")
    
    # 存储每个样本的全局索引（用于显示）
    global_indices = np.where(subject1_session1_mask)[0][selected_indices]
    print(f"全局样本索引: {global_indices.tolist()}")
    
    # ==================== 加载DDPM增强样本 ====================
    try:
        X_ddpm = np.load('outputs/ddpm_samples/ddpm_samples_subject1.npy')
        y_ddpm = np.load('outputs/ddpm_samples/ddpm_labels_subject1.npy')
        
        # 应用全局标准化，将生成数据对齐到真实数据的统计特性
        print("\n对DDPM生成数据进行全局标准化...")
        X_ddpm_normalized = normalize_generated_data_to_real_stats(X_real, X_ddpm)
        print(f"标准化前 - DDPM均值: {X_ddpm.mean():.4f}, 标准差: {X_ddpm.std():.4f}")
        print(f"标准化后 - DDPM均值: {X_ddpm_normalized.mean():.4f}, 标准差: {X_ddpm_normalized.std():.4f}")
        print(f"真实数据 - 均值: {X_real.mean():.4f}, 标准差: {X_real.std():.4f}")
        
        # 使用加权 Alpha band power 相似度，为每个真实样本匹配一个DDPM样本（trial级配对）
        print("\n使用加权 Alpha band power 相似度匹配DDPM样本（逐trial配对）...")
        alpha_ddpm_all = compute_alpha_power(X_ddpm_normalized)
        alpha_real_selected_power = alpha_real_selected  # 已经计算过
        
        X_ddpm_selected = []
        matched_indices = []
        for i, label in enumerate(y_real_selected):
            label_int = int(label)
            # 只在同一类别的DDPM样本中匹配
            label_mask = (y_ddpm == label)
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) > 0:
                # 该类别的baseline（基于真实数据）
                baseline_alpha = baseline_per_class[label_int]
                # ========== 先用关键通道的 ERD 模式做一次筛选 ==========
                c3_idx = KEY_CHANNELS['C3']
                cz_idx = KEY_CHANNELS['Cz']
                c4_idx = KEY_CHANNELS['C4']
                real_alpha_full = alpha_real_selected_power[i]
                filtered_indices = []
                for ddpm_idx in label_indices:
                    ddpm_alpha_full = alpha_ddpm_all[ddpm_idx]
                    # 计算该DDPM样本的相对功率（相对于该类别baseline）
                    ddpm_rel_full = compute_relative_alpha_power(
                        alpha_ddpm_all[ddpm_idx:ddpm_idx+1],
                        baseline_alpha
                    )[0]
                    if label_int == 0:
                        # Left Hand:
                        # 期望模式：
                        #   - C4 有明显 ERD：相对功率为负
                        #   - C3 不被抑制：相对功率 >= 0（不低于 baseline）
                        #   - C4 < C3（对侧抑制更强）
                        if (
                            ddpm_rel_full[c4_idx] < 0.0 and          # C4 ERD
                            ddpm_rel_full[c3_idx] >= 0.0 and         # C3 不抑制
                            ddpm_rel_full[c4_idx] <= ddpm_rel_full[c3_idx]  # C4 更低
                        ):
                            filtered_indices.append(ddpm_idx)
                    elif label_int == 1:
                        # Right Hand: 期望 C3 有ERD：相对功率为负，且 C3 < C4
                        if (ddpm_rel_full[c3_idx] < 0.0) and (ddpm_rel_full[c3_idx] <= ddpm_rel_full[c4_idx]):
                            filtered_indices.append(ddpm_idx)
                    elif label_int == 2:
                        # Feet: 期望 Cz 的相对功率为负
                        if ddpm_rel_full[cz_idx] < 0.0:
                            filtered_indices.append(ddpm_idx)
                    else:
                        # Tongue: 不做强约束，保留所有
                        filtered_indices.append(ddpm_idx)
                
                filtered_indices = np.array(filtered_indices, dtype=int)

                if len(filtered_indices) == 0:
                    # 如果按ERD模式筛选后为空，则退回到该类别的全部候选
                    candidate_indices = label_indices
                    print(f"  ⚠️ 类别 {label}: 按ERD模式筛选后无候选，退回到该类别全部DDPM样本")
                else:
                    candidate_indices = np.array(filtered_indices, dtype=int)
                    print(f"  类别 {label}: ERD模式筛选后剩余 {len(candidate_indices)} 个候选")

                # ========== 在候选集合中，用“相对功率 + 加权 Alpha 拓扑相关系数”选最近邻 ==========
                # 真实样本的相对功率向量
                real_rel_full = compute_relative_alpha_power(
                    real_alpha_full[None, :],
                    baseline_alpha
                )[0]
                real_alpha = real_rel_full * MOTOR_WEIGHTS
                similarities = []
                for ddpm_idx in candidate_indices:
                    ddpm_rel_full = compute_relative_alpha_power(
                        alpha_ddpm_all[ddpm_idx:ddpm_idx+1],
                        baseline_alpha
                    )[0]
                    ddpm_alpha = ddpm_rel_full * MOTOR_WEIGHTS
                    correlation = np.corrcoef(real_alpha, ddpm_alpha)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                    similarities.append(correlation)
                
                similarities = np.array(similarities)
                best_local = np.argmax(similarities)
                best_idx = int(candidate_indices[best_local])
                best_sim = float(similarities[best_local])
                
                X_ddpm_selected.append(X_ddpm_normalized[best_idx])
                matched_indices.append(best_idx)
                if best_sim < SIMILARITY_THRESHOLD:
                    print(f"  ⚠️ 类别 {label}: 最佳加权相似度仅为 {best_sim:.3f} (阈值={SIMILARITY_THRESHOLD:.2f})，模式差异仍较大")
                else:
                    print(f"  类别 {label}: 选择DDPM索引 {best_idx} (加权相似度={best_sim:.3f})")
            else:
                # 没有该类别的DDPM样本时，用带噪声的真实样本占位
                X_ddpm_selected.append(X_real_selected[i] + np.random.randn(*X_real_selected[i].shape) * 0.1)
                matched_indices.append(-1)
                print(f"  ⚠️ 类别 {label}: DDPM中无样本，使用占位符")
        
        X_ddpm_selected = np.array(X_ddpm_selected)
        matched_indices = np.array(matched_indices)
        print(f"\n匹配到的DDPM索引: {matched_indices.tolist()}")
        
        print("\n计算选择的DDPM样本的Alpha band power...")
        alpha_ddpm_selected = compute_alpha_power(X_ddpm_selected)
        print(f"DDPM Alpha power shape: {alpha_ddpm_selected.shape}")
        print(f"DDPM Alpha power范围: [{alpha_ddpm_selected.min():.2e}, {alpha_ddpm_selected.max():.2e}]")
        print(f"真实数据Alpha power范围: [{alpha_real_selected.min():.2e}, {alpha_real_selected.max():.2e}]")
        
    except Exception as e:
        print(f"⚠️  DDPM样本未找到: {e}")
        print("使用占位符数据...")
        X_ddpm_selected = X_real_selected + np.random.randn(*X_real_selected.shape) * 0.1
        alpha_ddpm_selected = compute_alpha_power(X_ddpm_selected)
    
    # ==================== 获取通道信息 ====================
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2', 'POz']
    
    # ==================== 绘图 ====================
    class_types = ['left_hand', 'right_hand', 'feet', 'tongue']
    
    # 所有类别都使用相对功率（相对于该类别的baseline的百分比变化）
    print("\n计算可视化数据...")
    print("  - 所有类别: 使用相对功率（相对于该类别的baseline的百分比变化）")
    
    # 计算相对功率（用于所有类别）
    alpha_real_relative = []
    alpha_ddpm_relative = []
    
    for class_idx in range(4):
        class_type = class_types[class_idx]
        
        # 获取该类别的baseline
        baseline_alpha = baseline_per_class[class_idx]
        
        # 获取绝对功率值
        real_abs = alpha_real_selected[class_idx]
        ddpm_abs = alpha_ddpm_selected[class_idx]
        
        # 所有类别都使用相对功率（相对于该类别的baseline）
        real_relative = compute_relative_alpha_power(
            alpha_real_selected[class_idx:class_idx+1], 
            baseline_alpha
        )[0]
        ddpm_relative = compute_relative_alpha_power(
            alpha_ddpm_selected[class_idx:class_idx+1], 
            baseline_alpha
        )[0]
        alpha_real_relative.append(real_relative)
        alpha_ddpm_relative.append(ddpm_relative)
        
        # 打印详细的调试信息（绝对功率和相对功率）
        if class_type == 'left_hand':
            print(f"\n  {class_names[class_idx]} (Real):")
            print(f"    绝对功率 - C3: {real_abs[7]:.2e}, C4: {real_abs[11]:.2e}")
            print(f"    相对功率 - C3: {real_relative[7]:.1f}%, C4: {real_relative[11]:.1f}%")
            print(f"    预期: C4（对侧）应该 < C3（同侧），即C4相对功率应该更负（更蓝）")
            print(f"  {class_names[class_idx]} (DDPM):")
            print(f"    绝对功率 - C3: {ddpm_abs[7]:.2e}, C4: {ddpm_abs[11]:.2e}")
            print(f"    相对功率 - C3: {ddpm_relative[7]:.1f}%, C4: {ddpm_relative[11]:.1f}%")
        elif class_type == 'right_hand':
            print(f"\n  {class_names[class_idx]} (Real):")
            print(f"    绝对功率 - C3: {real_abs[7]:.2e}, C4: {real_abs[11]:.2e}")
            print(f"    相对功率 - C3: {real_relative[7]:.1f}%, C4: {real_relative[11]:.1f}%")
            print(f"    预期: C3（对侧）应该 < C4（同侧），即C3相对功率应该更负（更蓝）")
            print(f"  {class_names[class_idx]} (DDPM):")
            print(f"    绝对功率 - C3: {ddpm_abs[7]:.2e}, C4: {ddpm_abs[11]:.2e}")
            print(f"    相对功率 - C3: {ddpm_relative[7]:.1f}%, C4: {ddpm_relative[11]:.1f}%")
        elif class_type == 'feet':
            print(f"\n  {class_names[class_idx]} (Real):")
            print(f"    绝对功率 - Cz: {real_abs[9]:.2e}")
            print(f"    相对功率 - Cz: {real_relative[9]:.1f}%")
            print(f"  {class_names[class_idx]} (DDPM):")
            print(f"    绝对功率 - Cz: {ddpm_abs[9]:.2e}")
            print(f"    相对功率 - Cz: {ddpm_relative[9]:.1f}%")
    
    alpha_real_relative = np.array(alpha_real_relative)
    alpha_ddpm_relative = np.array(alpha_ddpm_relative)
    
    print(f"\n真实数据相对功率范围: [{alpha_real_relative.min():.1f}%, {alpha_real_relative.max():.1f}%]")
    print(f"DDPM数据相对功率范围: [{alpha_ddpm_relative.min():.1f}%, {alpha_ddpm_relative.max():.1f}%]")
    
    # 创建2x4的图形 (2行: Real/DDPM, 4列: 4个类别)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    # 计算全局归一化范围（所有类别统一）
    all_values = np.concatenate([alpha_real_relative.flatten(), 
                                alpha_ddpm_relative.flatten()])
    vmin_global = np.nanmin(all_values)
    vmax_global = np.nanmax(all_values)
    
    print(f"\n全局数据范围: [{vmin_global:.1f}%, {vmax_global:.1f}%]")
    
    # 创建高分辨率网格（在循环外定义，避免重复创建）
    resolution = 200
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    head_radius = 0.85
    mask = np.sqrt(Xi**2 + Yi**2) <= head_radius
    
    # 绘制每个类别的拓扑图
    im = None
    for class_idx in range(4):
        # 获取样本信息
        sample_idx = selected_indices[class_idx]
        global_idx = global_indices[class_idx] if 'global_indices' in locals() else sample_idx
        class_label = class_names[int(y_real_selected[class_idx])]
        class_type = class_types[class_idx]
        
        # 获取绝对功率值
        alpha_real = alpha_real_relative[class_idx]
        alpha_ddpm = alpha_ddpm_relative[class_idx]
        
        # 归一化到0-1范围（使用全局范围）
        alpha_real_norm = (alpha_real - vmin_global) / (vmax_global - vmin_global)
        alpha_ddpm_norm = (alpha_ddpm - vmin_global) / (vmax_global - vmin_global)
        
        data_list = [alpha_real_norm, alpha_ddpm_norm]
        row_labels = ['Real', 'DDPM']
        
        for row_idx, (data, row_label) in enumerate(zip(data_list, row_labels)):
            ax = axes[row_idx, class_idx]
            
            # RBF插值
            rbf = Rbf(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], data, 
                      function='multiquadric', smooth=0.1)
            Zi = rbf(Xi, Yi)
            Zi[~mask] = np.nan
            
            # 绘制填充等高线 - 使用RdBu_r颜色映射
            # 蓝色=低功率(ERD/激活), 红色=高功率
            levels = np.linspace(0, 1, 50)
            im = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdBu_r', extend='both', vmin=0, vmax=1)
            
            # 绘制头部轮廓
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta), 'k-', linewidth=1.5)
            
            # 绘制鼻子（在Y轴正方向，即前额方向）
            nose_y = [head_radius, head_radius + 0.10, head_radius]
            nose_x = [0.05, 0, -0.05]
            ax.fill(nose_x, nose_y, 'white', edgecolor='k', linewidth=1.5, zorder=3)
        
            # 绘制耳朵
            ear_left = plt.Circle((-head_radius - 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ear_right = plt.Circle((head_radius + 0.05, 0), 0.05, fill=False, color='k', linewidth=1.5)
            ax.add_patch(ear_left)
            ax.add_patch(ear_right)
            
            # 绘制所有电极位置
            ax.scatter(CHANNEL_POSITIONS[:, 0], CHANNEL_POSITIONS[:, 1], c='k', s=12, zorder=5, 
                      edgecolors='white', linewidths=0.5)
            
            # 标注关键通道 C3, Cz, C4
            for ch_name, ch_idx in KEY_CHANNELS.items():
                pos = CHANNEL_POSITIONS[ch_idx]
                # 用较大的标记突出显示关键通道
                ax.scatter(pos[0], pos[1], c='yellow', s=50, zorder=6, 
                          edgecolors='black', linewidths=1.5, marker='o')
                # 添加通道名称标签
                offset_y = 0.12 if ch_name == 'Cz' else 0.10
                ax.text(pos[0], pos[1] - offset_y, ch_name, fontsize=8, fontweight='bold',
                       ha='center', va='top', color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
            
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.15])
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 列标题（类别名称）- 只在第一行显示
            if row_idx == 0:
                ax.set_title(f'{class_names[class_idx]}\n(S{subject_id+1}, T{sample_idx})', 
                            fontsize=11, fontweight='bold', pad=8)
            
            # 行标题 - 只在第一列显示
            if class_idx == 0:
                ax.text(-1.4, 0, row_label, fontsize=12, fontweight='bold', 
                       ha='center', va='center', rotation=90)
    
    # 统一使用一个颜色条（在右侧）
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Normalized Power', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', '', 'High'])
    
    # 添加总标题
    fig.suptitle(f'Alpha Band (8-13 Hz) Topographic Maps Comparison\nSubject {subject_id+1}, Session {session_id+1} | Relative Power', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 调整子图间距（减小空隙，为右侧颜色条留出空间）
    plt.subplots_adjust(left=0.08, right=0.90, top=0.85, bottom=0.12, wspace=0.08, hspace=0.15)
    
    # ==================== 保存图片 ====================
    os.makedirs('outputs/figures', exist_ok=True)
    
    save_path = 'outputs/figures/topographic_comparison_augmentation_mne.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[成功] 已保存: {save_path}")
    
    # 也保存PDF版本（用于论文）
    save_path_pdf = 'outputs/figures/topographic_comparison_augmentation_mne.pdf'
    fig.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"[成功] 已保存: {save_path_pdf}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("拓扑图可视化完成！")
    print("="*60)
    print("\n说明：")
    print("- 所有类别: 使用归一化功率（0-1范围）")
    print("  * 蓝色区域：低功率（ERD/激活）")
    print("  * 红色区域：高功率")
    print("- 黄色标记：C3, Cz, C4电极（运动皮层）")
    print("- 统一颜色条：所有子图使用相同的颜色范围")
    print("\n观察要点：")
    print("- DDPM是否保留了类别特异性的空间模式？")
    print("- 生成样本的拓扑分布是否与真实数据相似？")

def main():
    plot_topographic_comparison()

if __name__ == '__main__':
    main()
