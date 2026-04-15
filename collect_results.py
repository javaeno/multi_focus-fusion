# 创建一个整理实验结果的脚本 collect_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取你的评估结果
df = pd.read_csv('Lytro_evaluation.csv')
print("完整20张图结果：")
print(df)

# 计算统计指标
print("\n" + "="*50)
print("统计结果：")
print(f"平均 PSNR: {df['PSNR'].mean():.4f} ± {df['PSNR'].std():.4f} dB")
print(f"平均 SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
print(f"平均 Entropy: {df['Entropy'].mean():.4f} ± {df['Entropy'].std():.4f}")
print(f"PSNR范围: [{df['PSNR'].min():.4f}, {df['PSNR'].max():.4f}]")
print(f"SSIM范围: [{df['SSIM'].min():.4f}, {df['SSIM'].max():.4f}]")

# 绘制结果分布图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['PSNR'], bins=10, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('PSNR (dB)')
axes[0].set_ylabel('频数')
axes[0].set_title('PSNR分布')
axes[0].axvline(df['PSNR'].mean(), color='r', linestyle='--', label=f'均值={df["PSNR"].mean():.2f}')
axes[0].legend()

axes[1].hist(df['SSIM'], bins=10, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('SSIM')
axes[1].set_ylabel('频数')
axes[1].set_title('SSIM分布')
axes[1].axvline(df['SSIM'].mean(), color='r', linestyle='--', label=f'均值={df["SSIM"].mean():.4f}')
axes[1].legend()

axes[2].hist(df['Entropy'], bins=10, edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Entropy')
axes[2].set_ylabel('频数')
axes[2].set_title('信息熵分布')
axes[2].axvline(df['Entropy'].mean(), color='r', linestyle='--', label=f'均值={df["Entropy"].mean():.2f}')
axes[2].legend()

plt.tight_layout()
plt.savefig('results_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存统计结果到文本文件
with open('statistical_results.txt', 'w') as f:
    f.write("多聚焦图像融合实验结果统计\n")
    f.write("="*50 + "\n")
    f.write(f"测试图像数: {len(df)}\n\n")
    f.write(f"PSNR (dB):\n")
    f.write(f"  均值: {df['PSNR'].mean():.4f}\n")
    f.write(f"  标准差: {df['PSNR'].std():.4f}\n")
    f.write(f"  最小值: {df['PSNR'].min():.4f}\n")
    f.write(f"  最大值: {df['PSNR'].max():.4f}\n\n")
    f.write(f"SSIM:\n")
    f.write(f"  均值: {df['SSIM'].mean():.4f}\n")
    f.write(f"  标准差: {df['SSIM'].std():.4f}\n")
    f.write(f"  最小值: {df['SSIM'].min():.4f}\n")
    f.write(f"  最大值: {df['SSIM'].max():.4f}\n\n")
    f.write(f"Entropy:\n")
    f.write(f"  均值: {df['Entropy'].mean():.4f}\n")
    f.write(f"  标准差: {df['Entropy'].std():.4f}\n")
    f.write(f"  最小值: {df['Entropy'].min():.4f}\n")
    f.write(f"  最大值: {df['Entropy'].max():.4f}\n")