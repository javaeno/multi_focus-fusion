# -*- coding: utf-8 -*-
"""
测试自己训练的模型
"""
from torchvision import transforms
from skimage.io import imread
from MFF2 import MFF
import os
from network import Net
import torch
from utils import mkdir

# 加载自己训练的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(num_blocks=8, num_feat=128).to(device)

# 选择你训练好的权重
best_model_path = 'xx-xx-xx/best_net.pth'  # 替换成你的路径
net.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"加载模型: {best_model_path}")

# 测试集 - 用后10张
test_image1 = 'test_set/Lytro'
save_path = 'save_path_trained'
mkdir(os.path.join(save_path, 'Lytro'))

to_tensor = transforms.ToTensor()

print("\n开始测试自己训练的模型...")
for j in range(11, 21):  # 测试11-20
    index = '0' + str(j) if j <= 9 else str(j)

    print(f"\n处理第 {j} 对图像...")
    a = to_tensor(imread(os.path.join(test_image1, f'lytro-{index}-A.jpg')))[None, :, :, :].to(device)
    b = to_tensor(imread(os.path.join(test_image1, f'lytro-{index}-B.jpg')))[None, :, :, :].to(device)

    with torch.no_grad():
        net.eval()
        output = net(a, b)

    theta = output.detach().squeeze()
    theta[theta > 0.5] = 1
    theta[theta <= 0.5] = 0

    _input = torch.cat((a, b), 0)
    post_process = MFF(_input.cpu(), map_mode=theta.cpu())
    post_process.train(a.cpu() * theta.cpu() + b.cpu() * (1 - theta.cpu()))

    post_process.save_image(os.path.join(save_path, 'Lytro', f'{j}_fusion.png'))
    post_process.save_map(os.path.join(save_path, 'Lytro', f'{j}_map.png'))

    print(f"第 {j} 张完成")

print(f"\n完成！结果保存在: {save_path}")