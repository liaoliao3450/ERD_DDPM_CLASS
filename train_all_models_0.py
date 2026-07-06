#!/usr/bin/env python3
"""
批量训练所有生成模型的脚本
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_training(script_name, args_dict, output_dir):
    """运行单个训练脚本"""
    cmd = [sys.executable, script_name]
    
    # 添加参数
    for key, value in args_dict.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*60}")
    print(f"开始训练: {script_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {script_name} 训练完成! 耗时: {duration/60:.1f} 分钟")
        
        # 保存日志
        log_file = os.path.join(output_dir, f"{script_name.replace('.py', '')}_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"耗时: {duration/60:.1f} 分钟\n")
            f.write(f"返回码: {result.returncode}\n\n")
            f.write("标准输出:\n")
            f.write(result.stdout)
            f.write("\n标准错误:\n")
            f.write(result.stderr)
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ {script_name} 训练失败! 错误码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        
        # 保存错误日志
        log_file = os.path.join(output_dir, f"{script_name.replace('.py', '')}_error.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"错误码: {e.returncode}\n")
            f.write(f"耗时: {duration/60:.1f} 分钟\n\n")
            f.write("标准输出:\n")
            f.write(e.stdout or "无输出")
            f.write("\n标准错误:\n")
            f.write(e.stderr or "无错误信息")
        
        return False, duration


def main():
    parser = argparse.ArgumentParser(description='批量训练所有生成模型')
    
    # 通用参数
    parser.add_argument('--dataset_key', type=str, default='bci2a', 
                       choices=['bci2a', 'bci2b', 'seed'], help='数据集选择')
    parser.add_argument('--subject', type=str, default='A01E', help='特定被试')
    parser.add_argument('--window_size', type=int, default=750, help='窗口大小')
    parser.add_argument('--step', type=int, default=100, help='滑动步长')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--n_samples', type=int, default=500, help='生成样本数')
    
    # 控制参数
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['ddpm', 'vae', 'gan'],
                       choices=['ddpm', 'vae', 'gan', 'timegan', 'diffwave'],
                       help='要训练的模型列表')
    parser.add_argument('--mode', type=str, default='train_and_sample',
                       choices=['train', 'sample', 'train_and_sample'],
                       help='运行模式')
    parser.add_argument('--output_dir', type=str, default='./batch_training_results',
                       help='批量训练结果目录')
    parser.add_argument('--no_sliding', action='store_true', help='禁用滑动窗口')
    parser.add_argument('--conditional', action='store_true', help='条件生成')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    
    # 定义模型脚本映射
    model_scripts = {
        'ddpm': 'train_ddpm_simple.py',
        'vae': 'train_vae_simple.py', 
        'gan': 'train_gan_simple.py',
        # 可以添加更多模型
    }
    
    # 定义模型特定参数
    model_params = {
        'ddpm': {
            'n_steps': 1000,
            'sample_steps': 500,
            'base': 64,
            'lr': 2e-4
        },
        'vae': {
            'latent_dim': 64,
            'beta': 1.0,
            'lr': 1e-3,
            'cond_embed_dim': 32
        },
        'gan': {
            'z_dim': 100,
            'hidden': 64,
            'lr': 2e-4
        }
    }
    
    # 公共参数
    common_params = {
        'mode': args.mode,
        'dataset_key': args.dataset_key,
        'subject': args.subject,
        'window_size': args.window_size,
        'step': args.step,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'n_samples': args.n_samples,
        'no_sliding': args.no_sliding,
        'conditional': args.conditional
    }
    
    # 开始批量训练
    print(f"开始批量训练 {len(args.models)} 个模型...")
    print(f"模型列表: {', '.join(args.models)}")
    print(f"数据集: {args.dataset_key}, 被试: {args.subject}")
    print(f"模式: {args.mode}")
    
    results = {}
    total_start_time = time.time()
    
    for model_name in args.models:
        if model_name not in model_scripts:
            print(f"⚠️  跳过未支持的模型: {model_name}")
            continue
        
        script_name = model_scripts[model_name]
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"⚠️  脚本不存在: {script_path}")
            continue
        
        # 合并参数
        params = common_params.copy()
        params.update(model_params.get(model_name, {}))
        
        # 设置模型特定的输出目录
        params['ckpt_dir'] = f'./checkpoints/{model_name}'
        params['out_dir'] = f'./outputs/{model_name}_samples'
        
        # 运行训练
        success, duration = run_training(str(script_path), params, args.output_dir)
        results[model_name] = {'success': success, 'duration': duration}
    
    # 总结报告
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("批量训练完成!")
    print(f"总耗时: {total_duration/60:.1f} 分钟")
    print(f"{'='*60}")
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        duration = result['duration'] / 60
        print(f"{model_name:12s}: {status:8s} (耗时: {duration:5.1f} 分钟)")
        
        if result['success']:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"\n成功训练: {len(successful_models)} 个模型")
    if successful_models:
        print(f"  - {', '.join(successful_models)}")
    
    if failed_models:
        print(f"训练失败: {len(failed_models)} 个模型")
        print(f"  - {', '.join(failed_models)}")
    
    # 保存总结报告
    summary_file = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"批量训练总结报告\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"训练参数:\n")
        for key, value in common_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\n总耗时: {total_duration/60:.1f} 分钟\n\n")
        
        f.write(f"训练结果:\n")
        for model_name, result in results.items():
            status = "成功" if result['success'] else "失败"
            duration = result['duration'] / 60
            f.write(f"  {model_name}: {status} (耗时: {duration:.1f} 分钟)\n")
        
        f.write(f"\n成功训练: {len(successful_models)} 个模型\n")
        f.write(f"训练失败: {len(failed_models)} 个模型\n")
    
    print(f"\n详细日志保存在: {args.output_dir}")


if __name__ == '__main__':
    main()