"""数据集配置（供 data_loader.py 和其他脚本使用）

提供 DATA_PATH 和 DATASET_CONFIG (默认 BCI2a，兼容 data_loader.py)，
以及 DATASETS 字典（包含所有数据集配置）。
"""
import os

# ============================================================================
# 默认数据集 (BCI2a) - 兼容 data_loader.py 的直接导入
# ============================================================================
DATA_PATH = 'data/processed/BCI2a'

DATASET_CONFIG = {
    'n_classes': 4,       # 类别数 (Left/Right/Feet/Tongue)
    'n_subjects': 9,      # 被试数
    'n_sessions': 2,      # 每个被试的会话数
    'n_channels': 22,     # 通道数
    'n_samples': 1000,    # 每个样本的时间点数
    'fs': 250,            # 采样率 (Hz)
    'c3_idx': 7,          # C3 通道索引
    'c4_idx': 11,         # C4 通道索引
    'trials_per_session': 288,  # 每个会话的试验数
}

# ============================================================================
# 所有数据集配置
# ============================================================================
DATASETS = {
    'bci2a': {
        'data_dir': 'data/processed/BCI2a',
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
        'fs': 250,
        'c3_idx': 7,
        'c4_idx': 11,
        'n_subjects': 9,
        'n_sessions': 2,
        'trials_per_session': 288,
        'classifier_epochs': 200,
        'classifier_batch_size': 32,
        'classifier_lr': 1e-3,
    },
    'bci2b': {
        'data_dir': 'data/processed/BCI2b',
        'channels': 3,
        'n_samples': 1000,
        'num_classes': 2,          # BCI2b 是二分类 (左/右)
        'fs': 250,
        'c3_idx': 0,
        'c4_idx': 2,
        'n_subjects': 9,
        'n_sessions': 2,
        'trials_per_session': None,  # BCI2b 样本不均匀，由 data_loader 动态解析
        'classifier_epochs': 200,
        'classifier_batch_size': 32,
        'classifier_lr': 1e-3,
    },
    'physionet': {
        'data_dir': 'data/processed/PhysioNetMI4C',
        'channels': 64,
        'n_samples': 640,
        'num_classes': 4,
        'fs': 160,
        'c3_idx': 7,
        'c4_idx': 11,
        'n_subjects': 109,
        'n_sessions': 3,             # R04/R06->0, R08/R10->1, R12/R14->2
        'trials_per_session': None,  # 由 data_loader 动态解析
        'classifier_epochs': 300,
        'classifier_batch_size': 32,
        'classifier_lr': 1e-3,
        'cross_subject_mode': 'lmso_10fold',  # 十折交叉验证 (而非 LOSO)
    },
}


def get_dataset_config(dataset='bci2a'):
    """获取指定数据集的配置

    Args:
        dataset: 数据集名称 ('bci2a', 'bci2b', 'physionet')

    Returns:
        配置字典
    """
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集: {dataset}，可选: {list(DATASETS.keys())}")
    return DATASETS[dataset]
