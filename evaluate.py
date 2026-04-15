import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy


def evaluate_fusion_results(result_dir='save_path', dataset='Lytro', num_images=12):
    """评估融合结果"""
    print(f"\n评估 {dataset} 数据集...")
    print(f"结果文件夹: {result_dir}/{dataset}")

    results = []
    # 修正MFFW的路径
    if dataset == 'MFFW':
        test_dir = 'test_set/MFFW2'
    else:
        test_dir = f'test_set/{dataset}'

    for idx in range(1, num_images + 1):
        try:
            # 读取源图像
            if dataset == 'Lytro':
                idx_str = f'0{idx}' if idx < 10 else str(idx)
                a_path = os.path.join(test_dir, f'lytro-{idx_str}-A.jpg')
                b_path = os.path.join(test_dir, f'lytro-{idx_str}-B.jpg')
                img_a = imread(a_path)
                img_b = imread(b_path)
            else:  # MFFW
                a_path = os.path.join(test_dir, str(idx), 'image1.tif')
                b_path = os.path.join(test_dir, str(idx), 'image2.tif')
                img_a = imread(a_path)
                img_b = imread(b_path)

            # 读取融合图像
            fusion_path = os.path.join(result_dir, dataset, f'{idx}_fusion.png')
            print(f"正在读取: {fusion_path}")  # 调试：打印路径
            fusion = imread(fusion_path)

            # 归一化并转灰度
            a_gray = rgb2gray(img_a.astype(np.float32) / 255.0)
            b_gray = rgb2gray(img_b.astype(np.float32) / 255.0)
            f_gray = rgb2gray(fusion.astype(np.float32) / 255.0)

            # 调试：检查图像是否相同
            if np.array_equal(a_gray, f_gray):
                print(f"⚠️ 警告：图像 {idx} 的融合图与源图A完全相同！")
            if np.array_equal(b_gray, f_gray):
                print(f"⚠️ 警告：图像 {idx} 的融合图与源图B完全相同！")

            # 打印图像统计信息，检查是否有效
            print(f"  源图A 均值: {a_gray.mean():.4f}, 标准差: {a_gray.std():.4f}")
            print(f"  源图B 均值: {b_gray.mean():.4f}, 标准差: {b_gray.std():.4f}")
            print(f"  融合图均值: {f_gray.mean():.4f}, 标准差: {f_gray.std():.4f}")

            # 计算指标
            psnr_val = (psnr(a_gray, f_gray, data_range=1.0) +
                        psnr(b_gray, f_gray, data_range=1.0)) / 2
            ssim_val = (ssim(a_gray, f_gray, data_range=1.0) +
                        ssim(b_gray, f_gray, data_range=1.0)) / 2
            entropy_val = shannon_entropy(f_gray)

            results.append({
                'Image': f'{dataset}-{idx}',
                'PSNR': round(psnr_val, 4),
                'SSIM': round(ssim_val, 4),
                'Entropy': round(entropy_val, 4)
            })
            print(f"  ✅ 图像 {idx} 评估成功: PSNR={psnr_val:.4f}")

        except FileNotFoundError:
            print(f"❌ 图片 {idx} 评估失败: 文件未找到 - {fusion_path}")
        except Exception as e:
            print(f"❌ 图片 {idx} 评估失败: {e}")

    # 计算平均值
    if results:
        df = pd.DataFrame(results)
        print(f"\n平均 PSNR: {df['PSNR'].mean():.4f} dB")
        print(f"平均 SSIM: {df['SSIM'].mean():.4f}")
        print(f"平均 Entropy: {df['Entropy'].mean():.4f}")

        # 保存结果
        df.to_csv(f'{dataset}_evaluation.csv', index=False)
        print(f"结果已保存到 {dataset}_evaluation.csv")

    return results


if __name__ == '__main__':
    # 评估你的CBAM结果
    # evaluate_fusion_results('save_path_cbam_test', 'Lytro', 20)
    # 如果你想评估原始结果，可以取消下面的注释
    evaluate_fusion_results('save_path', 'Lytro', 20)