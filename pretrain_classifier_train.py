#!/usr/bin/env python3
"""
预训练EEG分类器并保存

只需要运行一次，后续所有实验都可以加载这个预训练好的分类器
"""
import sys, os, torch, numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'core/models/ddpm')
sys.path.insert(0, 'utils')
from class_discriminative import EEGClassifier, pretrain_classifier
from data_loader import load_bci2a_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    """加载数据"""
    X, y, subjects, sessions = load_bci2a_data()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    print("="*70)
    print("🎓 预训练EEG分类器（只需运行一次）")
    print("="*70)
    print(f"设备: {DEVICE}\n")
    
    # 加载数据
    print("📁 加载数据...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 预训练分类器
    print("\n🎓 预训练分类器 (500 epochs)...")
    clf = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
    clf = pretrain_classifier(
        clf,
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        epochs=500,
        batch_size=64,
        lr=1e-3,
        device=DEVICE,
        verbose=True
    )
    
    # 保存分类器
    print("\n💾 保存预训练分类器...")
    os.makedirs('checkpoints', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': clf.state_dict(),
        'channels': 22,
        'n_samples': 1000,
        'num_classes': 4,
    }
    
    torch.save(checkpoint, 'checkpoints/pretrained_classifier.pt')
    print("  ✅ 保存到: checkpoints/pretrained_classifier.pt")
    
    # 测试加载
    print("\n🔍 测试加载...")
    clf_test = EEGClassifier(channels=22, n_samples=1000, num_classes=4).to(DEVICE)
    clf_test.load_state_dict(checkpoint['model_state_dict'])
    print("  ✅ 加载成功！")
    
    print("\n" + "="*70)
    print("✅ 预训练分类器已保存！")
    print("="*70)
    print("\n后续实验可以直接加载这个分类器，无需重复训练。")
    print("\n使用方法:")
    print("  checkpoint = torch.load('checkpoints/pretrained_classifier.pt')")
    print("  clf.load_state_dict(checkpoint['model_state_dict'])")

if __name__ == '__main__':
    main()
