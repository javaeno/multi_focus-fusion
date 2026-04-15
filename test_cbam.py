# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:12:12 2020
@author: win10
Modified: CBAM version test
"""
from torchvision import transforms
from skimage.io import imread
from MFF2 import MFF
import os
from network import NetWithCBAM  # 导入CBAM版本
import torch
from utils import mkdir


def to_np(x):
    x = x.squeeze()
    x = x.permute(1, 2, 0)
    return x.detach().cpu().numpy()


print("=" * 60)
print("MFF-SSIM + CBAM 测试版本")
print("=" * 60)

# 配置网络 - 使用CBAM版本
net = NetWithCBAM(num_blocks=8, num_feat=128, reduction=16)
print("✓ CBAM网络初始化成功")

# **建议添加部分预训练权重（这样效果会更好）**
try:
    pretrained_dict = torch.load('weight/block8_feat128.pth', map_location=torch.device('cpu'))
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print(f"✓ 成功加载 {len(pretrained_dict)}/{len(model_dict)} 层的预训练权重")
except:
    print("⚠️ 使用随机权重")

# 测试集和输出文件夹
test_image1 = 'test_set/Lytro'
test_image2 = 'test_set/MFFW2'
save_path = 'save_path_cbam_test'
mkdir(os.path.join(save_path, 'Lytro'))
mkdir(os.path.join(save_path, 'MFFW'))

to_tensor = transforms.ToTensor()

# 先测试1张图
print("\n开始测试CBAM网络（只跑第1张图）...")

for j in range(15, 21):  # ✅ 修改为1，2：跑第1张
    if j <= 9:
        index = '0' + str(j)
    else:
        index = str(j)

    print(f"\n处理 Lytro 第 {j} 对图像...")

    a = to_tensor(imread(os.path.join(test_image1, 'lytro-' + index + '-A.jpg')))[None, :, :, :]
    b = to_tensor(imread(os.path.join(test_image1, 'lytro-' + index + '-B.jpg')))[None, :, :, :]
    with torch.no_grad():
        net.eval()
        output = net(a, b)

    theta = output.detach().squeeze()
    theta[theta > 0.5] = 1
    theta[theta <= 0.5] = 0

    _input = torch.cat((a, b), 0)
    post_process = MFF(_input, map_mode=theta)
    post_process.train(a * theta + b * (1 - theta))

    post_process.save_image(os.path.join(save_path, 'Lytro', str(j) + '_fusion.png'))
    post_process.save_map(os.path.join(save_path, 'Lytro', str(j) + '_map.png'))

    print(f"✓ 第 {j} 张处理完成")

print("\n" + "=" * 60)
print(f"CBAM测试完成！结果保存在: {save_path}")
print("=" * 60)