"""
统计显著性检验 (Statistical Significance Test)
对9个被试的结果进行配对t检验
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def cohen_d(x, y):
    """计算Cohen's d效应量"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def paired_cohen_d(x, y):
    """计算配对样本的Cohen's d"""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def main():
    print("="*60)
    print("统计显著性检验")
    print("="*60)
    
    # 读取实验结果
    print("\n读取实验结果...")
    with open('outputs/results/baseline_all_scenarios.json', 'r') as f:
        baseline = json.load(f)
    
    with open('outputs/results/ddpm_all_scenarios.json', 'r') as f:
        ddpm = json.load(f)
    
    # 三个场景
    scenarios = ['within_subject', 'cross_session', 'cross_subject']
    scenario_names = ['Within-Subject', 'Cross-Session', 'Cross-Subject']
    
    results = {}
    
    for scenario, name in zip(scenarios, scenario_names):
        print(f"\n{'='*60}")
        print(f"{name} 场景")
        print(f"{'='*60}")
        
        baseline_scores = baseline[scenario]['per_subject']
        ddpm_scores = ddpm[scenario]['per_subject']
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(ddpm_scores, baseline_scores)
        
        # Cohen's d
        d = paired_cohen_d(ddpm_scores, baseline_scores)
        
        # Wilcoxon符号秩检验（非参数检验）
        w_stat, w_p_value = stats.wilcoxon(ddpm_scores, baseline_scores)
        
        # 计算置信区间
        diff = np.array(ddpm_scores) - np.array(baseline_scores)
        ci = stats.t.interval(0.95, len(diff)-1, 
                             loc=np.mean(diff), 
                             scale=stats.sem(diff))
        
        print(f"\nBaseline 平均: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
        print(f"DDPM 平均:     {np.mean(ddpm_scores):.4f} ± {np.std(ddpm_scores):.4f}")
        print(f"平均改进:      {np.mean(diff):+.4f}")
        
        print(f"\n配对t检验:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        print(f"\nWilcoxon符号秩检验:")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value:     {w_p_value:.4f} {'***' if w_p_value < 0.001 else '**' if w_p_value < 0.01 else '*' if w_p_value < 0.05 else 'ns'}")
        
        print(f"\n效应量 (Cohen's d): {d:.4f}")
        if abs(d) < 0.2:
            effect = "小"
        elif abs(d) < 0.5:
            effect = "中等"
        elif abs(d) < 0.8:
            effect = "大"
        else:
            effect = "非常大"
        print(f"  效应大小: {effect}")
        
        print(f"\n95% 置信区间: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        
        # 保存结果
        results[scenario] = {
            'scenario_name': name,
            'baseline_mean': float(np.mean(baseline_scores)),
            'baseline_std': float(np.std(baseline_scores)),
            'ddpm_mean': float(np.mean(ddpm_scores)),
            'ddpm_std': float(np.std(ddpm_scores)),
            'improvement': float(np.mean(diff)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'w_statistic': float(w_stat),
            'w_p_value': float(w_p_value),
            'cohen_d': float(d),
            'effect_size': effect,
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'baseline_scores': [float(x) for x in baseline_scores],
            'ddpm_scores': [float(x) for x in ddpm_scores]
        }
    
    # 保存结果
    output_file = 'outputs/results/paper_experiments/statistical_test.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 可视化
    print("\n生成可视化图表...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
        ax = axes[idx]
        data = results[scenario]
        
        # 箱线图
        positions = [1, 2]
        bp = ax.boxplot([data['baseline_scores'], data['ddpm_scores']], 
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        labels=['Baseline', 'DDPM'])
        
        # 设置颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加散点
        for i, scores in enumerate([data['baseline_scores'], data['ddpm_scores']]):
            x = np.random.normal(positions[i], 0.04, size=len(scores))
            ax.scatter(x, scores, alpha=0.5, s=50, color='black')
        
        # 添加连线（显示配对关系）
        for j in range(len(data['baseline_scores'])):
            ax.plot([positions[0], positions[1]], 
                   [data['baseline_scores'][j], data['ddpm_scores'][j]], 
                   'k-', alpha=0.2, linewidth=0.5)
        
        # 标题和标签
        ax.set_title(f"{name}\np={data['p_value']:.4f}, d={data['cohen_d']:.2f}", 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加显著性标记
        if data['p_value'] < 0.001:
            sig = '***'
        elif data['p_value'] < 0.01:
            sig = '**'
        elif data['p_value'] < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        y_max = max(max(data['baseline_scores']), max(data['ddpm_scores']))
        ax.text(1.5, y_max + 0.05, sig, ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_fig = 'outputs/figures/paper_experiments/statistical_test.png'
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_fig}")
    
    # 打印LaTeX表格
    print("\n" + "="*60)
    print("LaTeX表格代码:")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Statistical Significance Test Results}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Scenario & Baseline & DDPM & $p$-value & Cohen's $d$ \\\\")
    print("\\hline")
    for scenario, name in zip(scenarios, scenario_names):
        data = results[scenario]
        print(f"{name} & {data['baseline_mean']:.2f}$\\pm${data['baseline_std']:.2f} & "
              f"{data['ddpm_mean']:.2f}$\\pm${data['ddpm_std']:.2f} & "
              f"{data['p_value']:.4f} & {data['cohen_d']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == '__main__':
    main()
