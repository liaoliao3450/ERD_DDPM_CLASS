#!/usr/bin/env python3
"""
Per-Subject分类结果可视化（数据增强论文）

创建3个柱状图：
1. Within-Subject: 9个被试的准确率对比
2. Cross-Session: 9个被试的准确率对比  
3. Cross-Subject: 9个被试的准确率对比

每个图显示: Baseline (No Aug) vs DDPM vs Gaussian vs VAE vs GAN
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_per_subject_results():
    """绘制per-subject结果对比图"""
    
    # ==================== 数据准备 ====================
    # 真实实验数据（从outputs/results/*.json获取）
    # 数据来源：BCI Competition IV Dataset 2a (9 subjects)
    
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    
    # Within-Subject结果（5-fold交叉验证）
    # 数据来源：outputs/results/*_all_scenarios.json
    within_subject_results = {
        'Baseline (No Aug)': [86.21, 61.21, 95.69, 70.69, 51.72, 57.76, 80.17, 87.93, 87.93],
        'DDPM (Ours)': [84.48, 46.55, 95.69, 73.28, 56.03, 61.21, 87.93, 87.07, 90.52],
        'Gaussian Noise': [82.76, 62.93, 87.07, 67.24, 50.00, 50.86, 86.21, 82.76, 87.93],
        'VAE': [75.86, 58.62, 81.03, 61.21, 48.28, 59.48, 74.14, 81.90, 84.48],
        'GAN': [83.62, 61.21, 93.97, 63.79, 46.55, 53.45, 84.48, 87.07, 87.93],
    }
    
    # Cross-Session结果（Session T → Session E）
    # 数据来源：outputs/results/*_all_scenarios.json
    cross_session_results = {
        'Baseline (No Aug)': [70.83, 59.72, 77.43, 52.43, 43.75, 44.79, 79.17, 78.13, 68.40],
        'DDPM (Ours)': [75.69, 59.72, 83.33, 58.33, 43.75, 51.74, 71.18, 76.04, 74.65],
        'Gaussian Noise': [73.26, 59.38, 79.51, 47.57, 42.71, 50.00, 69.79, 80.21, 72.22],
        'VAE': [72.22, 54.51, 78.47, 52.08, 37.15, 46.18, 68.75, 73.61, 68.40],
        'GAN': [72.57, 59.72, 78.47, 53.13, 40.63, 51.39, 70.49, 74.31, 69.79],
    }
    
    # Cross-Subject结果（LOSO - Leave-One-Subject-Out）
    # 数据来源：outputs/results/*_all_scenarios.json
    cross_subject_results = {
        'Baseline (No Aug)': [67.01, 44.10, 70.14, 48.96, 32.99, 36.46, 51.74, 60.42, 53.47],
        'DDPM (Ours)': [64.58, 46.18, 75.69, 54.86, 35.76, 37.15, 58.68, 62.15, 51.39],
        'Gaussian Noise': [61.11, 47.22, 62.85, 43.40, 35.76, 32.29, 44.79, 64.58, 50.35],
        'VAE': [68.75, 44.10, 66.32, 48.26, 36.11, 40.63, 53.82, 65.97, 69.44],
        'GAN': [59.38, 47.22, 69.10, 40.63, 33.33, 37.15, 61.11, 62.85, 61.11],
    }
    
    # ==================== 绘图设置 ====================
    colors = {
        'Baseline (No Aug)': '#3498DB',  # 蓝色
        'DDPM (Ours)': '#E74C3C',  # 红色
        'Gaussian Noise': '#95A5A6',  # 灰色
        'VAE': '#9B59B6',  # 紫色
        'GAN': '#F39C12',  # 橙色
    }
    
    # ==================== 创建3个子图 ====================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    datasets = [
        (within_subject_results, 'Within-Subject Classification', axes[0]),
        (cross_session_results, 'Cross-Session Classification', axes[1]),
        (cross_subject_results, 'Cross-Subject (LOSO) Classification', axes[2])
    ]
    
    for results, title, ax in datasets:
        x = np.arange(len(subjects))
        width = 0.15  # 柱子宽度
        
        # 绘制每种方法的柱子
        for i, (method, accuracies) in enumerate(results.items()):
            offset = (i - 2) * width  # 居中对齐
            bars = ax.bar(x + offset, accuracies, width, 
                         label=method, color=colors[method], alpha=0.8)
            
            # 在DDPM的柱子上标注数值（可选）
            if method == 'DDPM (Ours)':
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=8)
        
        # 设置图表属性
        ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
        ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([40, 90])  # 根据实际数据调整
        
        # 添加平均线（可选）
        for method, accuracies in results.items():
            mean_acc = np.mean(accuracies)
            if method == 'DDPM (Ours)':
                ax.axhline(y=mean_acc, color=colors[method], 
                          linestyle='--', linewidth=1.5, alpha=0.5,
                          label=f'DDPM Mean: {mean_acc:.2f}%')
    
    plt.tight_layout()
    
    # ==================== 保存图片 ====================
    os.makedirs('outputs/figures', exist_ok=True)
    
    # 保存单独的图片
    for i, (results, title, ax) in enumerate(datasets):
        fig_single = plt.figure(figsize=(8, 6))
        ax_single = fig_single.add_subplot(111)
        
        x = np.arange(len(subjects))
        width = 0.15
        
        for j, (method, accuracies) in enumerate(results.items()):
            offset = (j - 2) * width
            bars = ax_single.bar(x + offset, accuracies, width, 
                                label=method, color=colors[method], alpha=0.8)
            
            if method == 'DDPM (Ours)':
                for bar in bars:
                    height = bar.get_height()
                    ax_single.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{height:.1f}',
                                  ha='center', va='bottom', fontsize=8)
        
        ax_single.set_xlabel('Subject', fontsize=12, fontweight='bold')
        ax_single.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
        ax_single.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(subjects)
        ax_single.legend(fontsize=10, loc='lower right')
        ax_single.grid(axis='y', alpha=0.3, linestyle='--')
        ax_single.set_ylim([40, 90])
        
        # 保存
        scenario_name = title.split()[0].lower()
        save_path = f'outputs/figures/per_subject_{scenario_name}_augmentation.png'
        fig_single.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path}")
        plt.close(fig_single)
    
    # 保存组合图
    combined_path = 'outputs/figures/per_subject_all_scenarios_augmentation.png'
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {combined_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Per-Subject可视化完成！")
    print("="*60)
    print("\n✅ 使用真实实验数据")
    print("数据来源：outputs/results/*_all_scenarios.json")
    print("\n实验配置：")
    print("1. Within-Subject: 每个被试的5-fold交叉验证")
    print("2. Cross-Session: 每个被试的Session T→E测试")
    print("3. Cross-Subject: 每个被试作为测试集的LOSO")
    print("\n平均准确率：")
    print(f"  Within-Subject: Baseline={np.mean(within_subject_results['Baseline (No Aug)']):.2f}%, "
          f"DDPM={np.mean(within_subject_results['DDPM (Ours)']):.2f}%")
    print(f"  Cross-Session: Baseline={np.mean(cross_session_results['Baseline (No Aug)']):.2f}%, "
          f"DDPM={np.mean(cross_session_results['DDPM (Ours)']):.2f}%")
    print(f"  Cross-Subject: Baseline={np.mean(cross_subject_results['Baseline (No Aug)']):.2f}%, "
          f"DDPM={np.mean(cross_subject_results['DDPM (Ours)']):.2f}%")

def main():
    print("="*60)
    print("Per-Subject分类结果可视化（数据增强）")
    print("="*60)
    
    plot_per_subject_results()

if __name__ == '__main__':
    main()
