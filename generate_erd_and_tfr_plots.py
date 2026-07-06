#!/usr/bin/env python3
"""
生成ERD/ERS可视化图（参考DiffEEGBooth TBME论文风格）

Fig 4 (eegdiff): ERD/ERS时变曲线 - C3/C4通道在Left/Right Hand MI下的ERD%
  布局: 2行(Left Hand MI, Right Hand MI) × 2列(C3, C4)
  每个子图: Real EEG vs DDPM Generated 两条曲线

Fig 5 (eegdiff): Alpha频段拓扑图 - Real vs DDPM生成的空间功率分布
  布局: 2行(Real, DDPM) × 2列(Left Hand, Right Hand)
  使用RdBu_r色图, 蓝色=低功率/ERD, 红色=高功率
"""
import sys
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
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_bci2a_data

# ==================== 配置 ====================
FS = 250
NPERSEG = 128

# BCI2a 22通道
C3_IDX, C4_IDX = 7, 11

# 10-20系统通道位置（归一化，Y正=前额/鼻子方向）
CHANNEL_POSITIONS = np.array([
    [0.0, 0.72],    # Fz
    [-0.39, 0.54],  # FC3
    [-0.17, 0.54],  # FC1
    [0.0, 0.54],    # FCz
    [0.17, 0.54],   # FC2
    [0.39, 0.54],   # FC4
    [-0.59, 0.18],  # C5
    [-0.39, 0.18],  # C3
    [-0.17, 0.18],  # C1
    [0.0, 0.18],    # Cz
    [0.17, 0.18],   # C2
    [0.39, 0.18],   # C4
    [0.59, 0.18],   # C6
    [-0.39, -0.18], # CP3
    [-0.17, -0.18], # CP1
    [0.0, -0.18],   # CPz
    [0.17, -0.18],  # CP2
    [0.39, -0.18],  # CP4
    [-0.17, -0.54], # P1
    [0.0, -0.54],   # Pz
    [0.17, -0.54],  # P2
    [0.0, -0.72],   # POz
])

CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


def load_raw_bci2a_subject(subject_id=1, session=0):
    """
    从原始GDF文件加载包含基线的BCI2a数据（用于ERD计算）

    现有预处理数据(tmin=0, tmax=4)不包含cue前基线段，
    无法正确计算ERD。此函数直接从GDF文件提取包含基线的epoch
    (tmin=-1.5s, tmax=4.0s)，前1.5s作为基线。

    Returns:
        X: EEG数据 [N, 22, T]，包含基线段，单位V
        y: 标签 [N]，0-indexed
    """
    sys.path.insert(0, str(PROJECT_ROOT / 'core' / 'data_processing'))
    from process_bci2a import BCICompetition4Set2A
    from common import apply_preprocessing, extract_segment_trial

    gdf_dir = Path('E:/data/BCICIV_2a')
    if not gdf_dir.exists():
        # fallback to processed data
        data_dir = PROJECT_ROOT / 'data' / 'processed' / 'BCI2a'
        subj_name = f'A{subject_id:02d}{"T" if session == 0 else "E"}'
        X = np.load(str(data_dir / subj_name / 'X.npy')).astype(np.float32)
        y = np.load(str(data_dir / subj_name / 'y.npy')).astype(int)
        y = y - y.min()
        return X, y

    fname = str(gdf_dir / f'A{subject_id:02d}{"T" if session == 0 else "E"}.gdf')
    cnt = BCICompetition4Set2A(fname).load()
    cnt = apply_preprocessing(cnt, l_freq=4.0, h_freq=30.0, notch_freq=50.0,
                              car=True, resample_freq=250.0)

    # 提取含基线的epoch: tmin=-1.5s, tmax=4.0s
    X, y = extract_segment_trial(cnt, tmin=-1.5, tmax=4.0)
    y = y - y.min()

    return X.astype(np.float32), y


def load_ddpm_samples(subject_id=1):
    output_dir = PROJECT_ROOT / 'outputs' / 'ddpm_samples'
    samples_path = output_dir / f'ddpm_samples_subject{subject_id}.npy'
    labels_path = output_dir / f'ddpm_labels_subject{subject_id}.npy'
    if samples_path.exists() and labels_path.exists():
        return np.load(str(samples_path)), np.load(str(labels_path))
    return None, None


def denormalize_samples(gen_samples):
    """
    将标准化后的生成数据反标准化回原始尺度
    使用与load_bci2a_data()相同的全量数据统计参数
    """
    data_dir = PROJECT_ROOT / 'data' / 'processed' / 'BCI2a'
    X_all = np.load(str(data_dir / 'X.npy')).astype(np.float32)
    mean = X_all.mean(axis=(0, 2), keepdims=True)
    std = X_all.std(axis=(0, 2), keepdims=True) + 1e-8
    return gen_samples * std + mean


def standardize_data_with_baseline(data, task_start_sample):
    """
    使用全量数据统计参数对含基线数据进行标准化
    统计参数仅从任务段计算（与load_bci2a_data()一致）
    
    Args:
        data: 含基线的原始数据 [N, 22, T]（含基线段+任务段）
        task_start_sample: 任务段起始采样点
    
    Returns:
        标准化后的数据 [N, 22, T]
    """
    data_dir = PROJECT_ROOT / 'data' / 'processed' / 'BCI2a'
    X_all = np.load(str(data_dir / 'X.npy')).astype(np.float32)
    mean = X_all.mean(axis=(0, 2), keepdims=True)
    std = X_all.std(axis=(0, 2), keepdims=True) + 1e-8
    return (data - mean) / std


def compute_hilbert_envelopes(data, fs, fmin, fmax, channel_idx):
    """Bandpass滤波 + Hilbert变换提取瞬时功率包络"""
    from scipy.signal import butter, filtfilt, hilbert
    nyq = fs / 2
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype='band')
    n_trials = data.shape[0]
    n_times = data.shape[2]
    envelopes = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        filtered = filtfilt(b, a, data[i, channel_idx])
        envelopes[i] = np.abs(hilbert(filtered)) ** 2
    return envelopes


def compute_time_varying_erd(data, fs, fmin, fmax, channel_idx,
                              window_sec=0.5, step_sec=0.05,
                              baseline_start=None, baseline_end=None,
                              task_start=None, ref_power=None,
                              envelopes=None):
    """
    计算时变ERD百分比曲线（Hilbert变换方法）

    ERDS(t) = (P(t) - P_baseline) / P_baseline * 100
    负值=ERD，正值=ERS
    """
    n_trials, n_channels, n_times = data.shape

    if baseline_start is None:
        baseline_start = 0
    if baseline_end is None:
        baseline_end = int(1.5 * fs)
    if task_start is None:
        task_start = baseline_end

    # 计算或使用预计算的包络
    if envelopes is None:
        envelopes = compute_hilbert_envelopes(data, fs, fmin, fmax, channel_idx)

    # 计算参考基线功率
    if ref_power is None:
        baseline_power = envelopes[:, baseline_start:baseline_end].mean()
    else:
        baseline_power = ref_power

    # 滑动窗口
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    erd_per_trial = []
    times = []
    start = task_start
    while start + window_samples <= n_times:
        t_center = (start + window_samples / 2 - task_start) / fs
        times.append(t_center)

        for i in range(n_trials):
            seg_power = envelopes[i, start:start + window_samples].mean()
            erd_val = (seg_power - baseline_power) / baseline_power * 100
            erd_per_trial.append(erd_val)

        start += step_samples

    n_windows = len(times)
    erd_per_trial = np.array(erd_per_trial).reshape(n_trials, n_windows)

    return np.array(times), erd_per_trial.mean(axis=0), erd_per_trial.std(axis=0)


def compute_ref_baseline_power_hilbert(data, fs, fmin, fmax, channel_idx,
                                        baseline_start=0, baseline_end=375):
    """
    使用Hilbert方法计算参考基线功率（与compute_time_varying_erd一致）
    """
    from scipy.signal import butter, filtfilt, hilbert

    nyq = fs / 2
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype='band')
    ch_data = data[:, channel_idx, :]  # [N, T]

    envelopes = np.zeros_like(ch_data)
    for i in range(data.shape[0]):
        filtered = filtfilt(b, a, ch_data[i])
        analytic = hilbert(filtered)
        envelopes[i] = np.abs(analytic) ** 2

    return envelopes[:, baseline_start:baseline_end].mean()


def compute_ref_baseline_power(data, fs, fmin, fmax, channel_idx,
                                baseline_start=0, baseline_end=375):
    """
    从含基线的数据中计算参考基线功率（所有试次平均）
    用于为无基线的生成数据提供ERD参考
    """
    bl_len = baseline_end - baseline_start
    all_bl_psd = []
    for i in range(data.shape[0]):
        bl_data = data[i, channel_idx, baseline_start:baseline_end]
        f_bl, psd_bl = welch(bl_data, fs=fs, nperseg=min(bl_len, NPERSEG))
        all_bl_psd.append(psd_bl)
    all_bl_psd = np.array(all_bl_psd)
    mean_bl_psd = all_bl_psd.mean(axis=0)
    mask = (f_bl >= fmin) & (f_bl <= fmax)
    return np.trapezoid(mean_bl_psd[mask], f_bl[mask]) if mask.sum() > 0 else 1e-10


def generate_erd_curves(real_data, gen_data, labels_real, labels_gen, fs, save_path,
                         baseline_start=0, baseline_end=375, task_start=375,
                         gen_task_start=0):
    """
    ERD/ERS时变曲线图

    布局: 2行(Left Hand MI, Right Hand MI) × 2列(C3, C4)
    每个子图: Real EEG vs DDPM Generated

    计算方法: STFT alpha band power + 校准基线
    - Real: pre-cue基线参考 (标准ERD方法, 侧化正确)
    - Gen: 校准基线参考 (借用Real的pre-cue/early-task比率k估计Gen的pre-cue基线)
    - k = real_pre_cue_ref / real_early_task_ref, gen_ref = gen_early * k
    - STFT提取alpha (8-13Hz)时频功率, 比 Hilbert 对生成数据更鲁棒
    - 1s移动平均平滑后计算ERD%
    - 标准差: per-trial ERD的跨试次std
    """
    from scipy.signal import butter, filtfilt, stft
    from scipy.ndimage import uniform_filter1d
    from scipy.interpolate import interp1d

    smooth_sec = 1.0       # 1s移动平均平滑
    ref_sec = 0.5          # early-task参考窗口: 前0.5s
    nperseg_stft = 128     # STFT窗口 (0.512s @ 250Hz)
    noverlap_stft = 120    # STFT重叠

    nyq = fs / 2
    b_bp, a_bp = butter(4, [8 / nyq, 13 / nyq], btype='band')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    show_classes = [0, 1]
    class_labels = ['Left-hand imagery', 'Right-hand imagery']
    colors = {'real': '#1f77b4', 'ddpm': '#d62728'}

    ref_samples = int(ref_sec * fs)

    def compute_stft_power(trial_sig, fs, b_bp, a_bp, nperseg, noverlap, t_orig):
        """STFT alpha band power for a single trial"""
        sig = filtfilt(b_bp, a_bp, trial_sig)
        f, t, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        alpha_mask = (f >= 8) & (f <= 13)
        power_t = np.sum(np.abs(Zxx[alpha_mask, :]) ** 2, axis=0)
        interp_func = interp1d(t, power_t, kind='linear', fill_value='extrapolate')
        return interp_func(t_orig)

    for row, class_id in enumerate(show_classes):
        mask_real = labels_real == class_id
        mask_gen = labels_gen == class_id

        if mask_real.sum() < 5 or mask_gen.sum() < 5:
            continue

        real_class = real_data[mask_real]
        gen_class = gen_data[mask_gen]

        for col, (ch_idx, ch_name) in enumerate([(C3_IDX, 'C3'), (C4_IDX, 'C4')]):
            ax = axes[row, col]

            # ---- Real: STFT alpha power (full: baseline + task) ----
            n_real = real_class.shape[0]
            t_orig_r = np.arange(real_class.shape[2]) / fs
            real_env = np.zeros((n_real, real_class.shape[2]))
            for i in range(n_real):
                real_env[i] = compute_stft_power(
                    real_class[i, ch_idx], fs, b_bp, a_bp,
                    nperseg_stft, noverlap_stft, t_orig_r)

            real_env_mean = real_env.mean(axis=0)  # [T_full]
            # Pre-cue基线参考 (标准方法)
            real_precue_ref = real_env_mean[baseline_start:baseline_end].mean()
            # Early-task参考 (用于校准gen)
            real_early_ref = real_env_mean[task_start:task_start + ref_samples].mean()
            # 校准系数: pre-cue / early-task
            k_calib = real_precue_ref / real_early_ref

            # ---- Gen: STFT alpha power ----
            n_gen = gen_class.shape[0]
            t_orig_g = np.arange(gen_class.shape[2]) / fs
            gen_env = np.zeros((n_gen, gen_class.shape[2]))
            for i in range(n_gen):
                gen_env[i] = compute_stft_power(
                    gen_class[i, ch_idx], fs, b_bp, a_bp,
                    nperseg_stft, noverlap_stft, t_orig_g)

            gen_env_mean = gen_env.mean(axis=0)  # [T_task]
            # Gen校准基线: early-task * k
            gen_early_ref = gen_env_mean[:ref_samples].mean()
            gen_ref = gen_early_ref * k_calib

            # ---- 移动平均平滑 + ERD计算 ----
            smooth_size = int(smooth_sec * fs)

            # Real: 只取task段, pre-cue参考
            real_task_env = real_env_mean[task_start:]
            real_task_smooth = uniform_filter1d(real_task_env, size=smooth_size)
            times_r = np.arange(len(real_task_smooth)) / fs
            erd_r_mean = (real_task_smooth - real_precue_ref) / real_precue_ref * 100

            # Real per-trial ERD for std
            erd_r_trials = np.zeros((n_real, len(real_task_smooth)))
            for i in range(n_real):
                trial_task = real_env[i, task_start:]
                trial_smooth = uniform_filter1d(trial_task, size=smooth_size)
                erd_r_trials[i] = (trial_smooth - real_precue_ref) / real_precue_ref * 100
            erd_r_std = erd_r_trials.std(axis=0)

            # Gen: 校准基线参考
            gen_smooth = uniform_filter1d(gen_env_mean, size=smooth_size)
            times_g = np.arange(len(gen_smooth)) / fs
            erd_g_mean = (gen_smooth - gen_ref) / gen_ref * 100

            # Gen per-trial ERD for std
            erd_g_trials = np.zeros((n_gen, len(gen_smooth)))
            for i in range(n_gen):
                trial_smooth = uniform_filter1d(gen_env[i], size=smooth_size)
                erd_g_trials[i] = (trial_smooth - gen_ref) / gen_ref * 100
            erd_g_std = erd_g_trials.std(axis=0)

            # ---- 绘图 ----
            ax.plot(times_r, erd_r_mean, '-', color=colors['real'], linewidth=2,
                    label='Real EEG')
            ax.fill_between(times_r, erd_r_mean - erd_r_std, erd_r_mean + erd_r_std,
                            color=colors['real'], alpha=0.15)

            ax.plot(times_g, erd_g_mean, '--', color=colors['ddpm'], linewidth=2,
                    label='DDPM Generated')
            ax.fill_between(times_g, erd_g_mean - erd_g_std, erd_g_mean + erd_g_std,
                            color=colors['ddpm'], alpha=0.15)

            ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
                       label='Cue onset')

            ax.set_ylabel('ERD/ERS (%)', fontsize=11)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f'Channel {ch_name}', fontsize=12, fontweight='bold')
            if row == 1:
                ax.set_xlabel('Time after cue (s)', fontsize=11)

            if col == 0:
                ax.annotate(class_labels[row], xy=(-0.2, 0.5), xycoords='axes fraction',
                            fontsize=11, fontweight='bold', ha='center', va='center', rotation=90)

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('ERD/ERS Curves (Alpha Band, 8-13 Hz)\nReal EEG vs DDPM Generated',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.06, 0, 1, 0.94])
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ERD curves saved to {save_path}")


def compute_alpha_power_topo(data, fs, fmin=8, fmax=13, task_start=0):
    """
    计算Alpha频段各通道的平均功率（用于拓扑图）
    与DiffEEGBooth Fig.5风格一致：展示功率空间分布，低功率=ERD
    对于生成数据（无基线），task_start=0即可
    """
    n_trials, n_channels, n_times = data.shape
    alpha_powers = np.zeros(n_channels)

    for ch in range(n_channels):
        powers = []
        for i in range(n_trials):
            f, psd = welch(data[i, ch, task_start:], fs=fs, nperseg=min(n_times - task_start, NPERSEG))
            mask = (f >= fmin) & (f <= fmax)
            powers.append(np.trapezoid(psd[mask], f[mask]) if mask.sum() > 0 else 0)
        alpha_powers[ch] = np.mean(powers)

    return alpha_powers


def compute_erd_topo(data, fs, fmin=8, fmax=13, baseline_start=0, baseline_end=375, task_start=375):
    """
    计算Alpha频段各通道的ERD百分比（用于拓扑图，STFT方法）
    ERD% = (P_task - P_baseline) / P_baseline * 100
    负值=ERD（功率降低），正值=ERS（功率增加）
    """
    from scipy.signal import butter, filtfilt, stft
    from scipy.interpolate import interp1d

    n_trials, n_channels, n_times = data.shape
    nyq = fs / 2
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype='band')
    t_orig = np.arange(n_times) / fs

    erd_powers = np.zeros(n_channels)

    for ch in range(n_channels):
        # STFT alpha band power
        envelopes = np.zeros((n_trials, n_times))
        for i in range(n_trials):
            sig = filtfilt(b, a, data[i, ch, :])
            f, t, Zxx = stft(sig, fs=fs, nperseg=128, noverlap=120)
            alpha_mask = (f >= fmin) & (f <= fmax)
            power_t = np.sum(np.abs(Zxx[alpha_mask, :]) ** 2, axis=0)
            interp_func = interp1d(t, power_t, kind='linear', fill_value='extrapolate')
            envelopes[i] = interp_func(t_orig)

        # 基线参考功率
        ref_power = envelopes[:, baseline_start:baseline_end].mean()
        # 任务段功率
        task_power = envelopes[:, task_start:].mean()

        erd_powers[ch] = (task_power - ref_power) / ref_power * 100

    return erd_powers


def compute_alpha_topo_zscore(data, fs, task_start=None, fmin=8, fmax=13):
    """
    Alpha频段功率的z-score空间分布（跨通道归一化）

    避免基线参考问题，直接比较alpha功率的空间分布模式。
    z < 0 表示该通道alpha功率低于全脑平均（即ERD更强）。
    """
    from scipy.signal import butter, filtfilt, stft
    from scipy.interpolate import interp1d

    n_trials, n_channels, n_times = data.shape
    nyq = fs / 2
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype='band')
    t_orig = np.arange(n_times) / fs

    alpha_power = np.zeros(n_channels)
    for ch in range(n_channels):
        env = np.zeros((n_trials, n_times))
        for i in range(n_trials):
            sig = filtfilt(b, a, data[i, ch, :])
            f, t, Zxx = stft(sig, fs=fs, nperseg=128, noverlap=120)
            alpha_mask = (f >= fmin) & (f <= fmax)
            power_t = np.sum(np.abs(Zxx[alpha_mask, :]) ** 2, axis=0)
            interp_func = interp1d(t, power_t, kind='linear', fill_value='extrapolate')
            env[i] = interp_func(t_orig)
        if task_start is not None:
            alpha_power[ch] = env[:, task_start:].mean()
        else:
            alpha_power[ch] = env.mean()

    # Z-score across channels
    z_scores = (alpha_power - alpha_power.mean()) / (alpha_power.std() + 1e-8)
    return z_scores


def interpolate_topo(values, positions, grid_res=200):
    xi = np.linspace(-1, 1, grid_res)
    yi = np.linspace(-1, 1, grid_res)
    xi, yi = np.meshgrid(xi, yi)
    rbfi = Rbf(positions[:, 0], positions[:, 1], values,
               function='multiquadric', smooth=0.1)
    zi = rbfi(xi, yi)
    r = np.sqrt(xi**2 + yi**2)
    zi[r > 0.85] = np.nan
    return xi, yi, zi


def plot_head_outline(ax, radius=0.85):
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-', linewidth=1.5)
    # 鼻子
    ax.plot([0, 0], [radius, radius + 0.1], 'k-', linewidth=1.5)
    # 左耳
    ax.plot([-radius - 0.1, -radius], [0, 0], 'k-', linewidth=1.5)
    # 右耳
    ax.plot([radius, radius + 0.1], [0, 0], 'k-', linewidth=1.5)


def generate_erd_topo(real_data, gen_data, labels_real, labels_gen, fs, save_path,
                       real_task_start=375, gen_task_start=0):
    """
    Alpha频段ERD拓扑图：z-score归一化的alpha功率空间分布

    布局: 2行(Real EEG, DDPM Generated) × 2列(Left Hand, Right Hand)
    使用RdBu_r色图: 蓝色=alpha功率低于平均(ERD), 红色=alpha功率高于平均(ERS)
    Z-score归一化避免基线参考问题，直接比较空间模式。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    show_classes = [0, 1]
    class_labels = ['Left Hand MI', 'Right Hand MI']

    # 计算z-score归一化的alpha功率空间分布
    all_zscore = {}
    for class_id in show_classes:
        mask_real = labels_real == class_id
        mask_gen = labels_gen == class_id

        real_class = real_data[mask_real]
        gen_class = gen_data[mask_gen]

        if mask_real.sum() > 0:
            all_zscore[('real', class_id)] = compute_alpha_topo_zscore(
                real_class, fs, task_start=real_task_start)

        if mask_gen.sum() > 0:
            all_zscore[('gen', class_id)] = compute_alpha_topo_zscore(
                gen_class, fs, task_start=None)

    # 统一色标范围（以0为中心对称，取所有数据的最大绝对值）
    all_vals = np.concatenate([v for v in all_zscore.values()])
    abs_max = max(abs(all_vals.min()), abs(all_vals.max()))

    im = None
    for col, class_id in enumerate(show_classes):
        key_real = ('real', class_id)
        key_gen = ('gen', class_id)

        # Row 0: Real EEG
        if key_real in all_zscore:
            z_real = all_zscore[key_real]
            xi, yi, zi = interpolate_topo(z_real, CHANNEL_POSITIONS)

            ax = axes[0, col]
            im = ax.pcolormesh(xi, yi, zi, cmap='RdBu_r',
                               vmin=-abs_max, vmax=abs_max, shading='auto')
            plot_head_outline(ax)
            ax.set_aspect('equal')
            ax.axis('off')

            ax.plot(CHANNEL_POSITIONS[C3_IDX, 0], CHANNEL_POSITIONS[C3_IDX, 1],
                    'ko', markersize=5, zorder=5)
            ax.plot(CHANNEL_POSITIONS[C4_IDX, 0], CHANNEL_POSITIONS[C4_IDX, 1],
                    'ko', markersize=5, zorder=5)
            ax.text(CHANNEL_POSITIONS[C3_IDX, 0] - 0.12,
                    CHANNEL_POSITIONS[C3_IDX, 1] + 0.08, 'C3',
                    fontsize=8, fontweight='bold', zorder=5)
            ax.text(CHANNEL_POSITIONS[C4_IDX, 0] + 0.05,
                    CHANNEL_POSITIONS[C4_IDX, 1] + 0.08, 'C4',
                    fontsize=8, fontweight='bold', zorder=5)

            if col == 0:
                ax.set_ylabel('Real EEG', fontsize=12, fontweight='bold')
            ax.set_title(class_labels[col], fontsize=11, fontweight='bold')

        # Row 1: DDPM Generated
        if key_gen in all_zscore:
            z_gen = all_zscore[key_gen]
            xi, yi, zi = interpolate_topo(z_gen, CHANNEL_POSITIONS)

            ax = axes[1, col]
            im = ax.pcolormesh(xi, yi, zi, cmap='RdBu_r',
                               vmin=-abs_max, vmax=abs_max, shading='auto')
            plot_head_outline(ax)
            ax.set_aspect('equal')
            ax.axis('off')

            ax.plot(CHANNEL_POSITIONS[C3_IDX, 0], CHANNEL_POSITIONS[C3_IDX, 1],
                    'ko', markersize=5, zorder=5)
            ax.plot(CHANNEL_POSITIONS[C4_IDX, 0], CHANNEL_POSITIONS[C4_IDX, 1],
                    'ko', markersize=5, zorder=5)
            ax.text(CHANNEL_POSITIONS[C3_IDX, 0] - 0.12,
                    CHANNEL_POSITIONS[C3_IDX, 1] + 0.08, 'C3',
                    fontsize=8, fontweight='bold', zorder=5)
            ax.text(CHANNEL_POSITIONS[C4_IDX, 0] + 0.05,
                    CHANNEL_POSITIONS[C4_IDX, 1] + 0.08, 'C4',
                    fontsize=8, fontweight='bold', zorder=5)

            if col == 0:
                ax.set_ylabel('DDPM Generated', fontsize=12, fontweight='bold')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Alpha Power (z-score)', fontsize=11)

    fig.suptitle('Alpha Band (8-13 Hz) Spatial Distribution\n(Blue=ERD/Desynchronization, Red=ERS/Synchronization)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ERD topo saved to {save_path}")


def main():
    output_dir = PROJECT_ROOT / 'paper' / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    BASELINE_END = 375   # 1.5s * 250Hz
    TASK_START = 375     # cue时刻

    print("Loading RAW BCI2a data with baseline from GDF files...")
    X_train, y_train = load_raw_bci2a_subject(subject_id=1, session=0)
    print(f"Raw data shape: {X_train.shape}")
    print(f"Raw data range: [{X_train.min()*1e6:.2f}, {X_train.max()*1e6:.2f}] uV")
    print(f"Labels: {np.bincount(y_train)}")

    print("\nStandardizing real data to match DDPM training space...")
    X_train_std = standardize_data_with_baseline(X_train, TASK_START)
    print(f"Standardized real data range: [{X_train_std.min():.2f}, {X_train_std.max():.2f}]")

    print("\nLoading DDPM generated samples...")
    gen_samples, gen_labels = load_ddpm_samples(subject_id=1)
    if gen_samples is None:
        print("ERROR: No DDPM samples found")
        return

    gen_labels = gen_labels.astype(int)
    if gen_samples.ndim == 4:
        gen_samples = gen_samples.squeeze(1)
    print(f"Gen samples range: [{gen_samples.min():.2f}, {gen_samples.max():.2f}]")
    print(f"Gen samples shape (task only): {gen_samples.shape}")

    # 生成数据已做per-channel标准化（与生成脚本一致）
    # 无需额外缩放，ERD计算使用各自内部参考

    print("\n1. Generating ERD/ERS time-varying curves (Fig.4 style)...")
    generate_erd_curves(X_train_std, gen_samples, y_train, gen_labels, FS,
                        output_dir / 'erd_ers_curves.pdf',
                        baseline_start=0, baseline_end=BASELINE_END,
                        task_start=TASK_START, gen_task_start=0)
    generate_erd_curves(X_train_std, gen_samples, y_train, gen_labels, FS,
                        output_dir / 'erd_ers_curves.png',
                        baseline_start=0, baseline_end=BASELINE_END,
                        task_start=TASK_START, gen_task_start=0)

    print("\n2. Generating ERD topographic maps (Fig.5 style)...")
    generate_erd_topo(X_train_std, gen_samples, y_train, gen_labels, FS,
                      output_dir / 'erd_maps_comparison.pdf',
                      real_task_start=TASK_START, gen_task_start=0)
    generate_erd_topo(X_train_std, gen_samples, y_train, gen_labels, FS,
                      output_dir / 'erd_maps_comparison.png',
                      real_task_start=TASK_START, gen_task_start=0)

    print("\nDone!")


if __name__ == '__main__':
    main()
