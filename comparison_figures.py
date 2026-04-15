# create_comparison_figures.py
import matplotlib.pyplot as plt
from skimage.io import imread
import os
import pandas as pd
import numpy as np

# 1. 先读取评估结果（这是你缺的关键步骤）
print("读取评估结果...")
df = pd.read_csv('Lytro_evaluation.csv')
print(f"找到 {len(df)} 张图像的结果")

# 2. 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 3. 选择要显示的图像（根据你的实际数据调整）
# 从你的数据看，有1-20张，我们选几张典型的
selected_images = [1, 5, 9, 15, 18]  # 选了5张不同场景的

# 4. 创建画布
fig, axes = plt.subplots(len(selected_images), 4, figsize=(20, 5 * len(selected_images)))

# 5. 循环读取和显示
for i, idx in enumerate(selected_images):
    try:
        # 格式化索引（1→01, 10→10）
        idx_str = f'0{idx}' if idx < 10 else str(idx)

        # 构建文件路径（根据你的项目结构调整）
        a_path = f'test_set/Lytro/lytro-{idx_str}-A.jpg'
        b_path = f'test_set/Lytro/lytro-{idx_str}-B.jpg'
        fusion_path = f'save_path/Lytro/{idx}_fusion.png'
        map_path = f'save_path/Lytro/{idx}_map.png'

        print(f"读取第 {idx} 张...")
        print(f"  A图: {a_path}")
        print(f"  B图: {b_path}")
        print(f"  融合图: {fusion_path}")
        print(f"  焦点图: {map_path}")

        # 读取图像
        a = imread(a_path)
        b = imread(b_path)
        fusion = imread(fusion_path)
        focus_map = imread(map_path)

        # 获取该图像的PSNR值（用于标题）
        psnr_value = df[df['Image'] == f'Lytro-{idx}']['PSNR'].values[0]

        # 显示
        # 源图A
        axes[i, 0].imshow(a)
        axes[i, 0].set_title(f'源图A (Lytro-{idx})', fontsize=12)
        axes[i, 0].axis('off')

        # 源图B
        axes[i, 1].imshow(b)
        axes[i, 1].set_title(f'源图B (Lytro-{idx})', fontsize=12)
        axes[i, 1].axis('off')

        # 融合结果
        axes[i, 2].imshow(fusion)
        axes[i, 2].set_title(f'融合结果\nPSNR={psnr_value:.2f}dB', fontsize=12)
        axes[i, 2].axis('off')

        # 焦点图
        axes[i, 3].imshow(focus_map, cmap='gray')
        axes[i, 3].set_title('焦点图\n(白:选A, 黑:选B)', fontsize=12)
        axes[i, 3].axis('off')

    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        # 如果文件不存在，显示空白
        for j in range(4):
            axes[i, j].text(0.5, 0.5, f'图像{idx}不存在',
                            ha='center', va='center')
            axes[i, j].axis('off')
    except Exception as e:
        print(f"❌ 其他错误: {e}")

# 6. 调整布局并保存
plt.tight_layout()
plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ 对比图已保存为: fusion_comparison.png")
plt.show()

# 7. 可选：打印统计信息
print("\n" + "=" * 50)
print("选择的图像及其PSNR值：")
for idx in selected_images:
    psnr = df[df['Image'] == f'Lytro-{idx}']['PSNR'].values[0]
    ssim = df[df['Image'] == f'Lytro-{idx}']['SSIM'].values[0]
    print(f"Lytro-{idx}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")