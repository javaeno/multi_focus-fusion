import torch.utils.data as Data
from glob import glob
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np


class Dataset(Data.Dataset):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root: 数据集根目录
            transform: 数据增强
            train: 是否为训练集
        """
        if train:
            # 训练集：用Lytro的前10张
            self.image_indices = list(range(1, 11))  # 1-10训练
            print(f"训练集使用 {len(self.image_indices)} 对图像")
        else:
            # 测试集：用Lytro的后10张
            self.image_indices = list(range(11, 21))  # 11-20测试
            print(f"测试集使用 {len(self.image_indices)} 对图像")

        self.root = root
        self._tensor = transforms.ToTensor()
        self.train = train

        # 添加灰度转换
        self.to_gray = transforms.Grayscale(num_output_channels=1)

        # 数据增强：训练时用
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
        ]) if train else None

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        idx = self.image_indices[index]

        # 格式化索引
        idx_str = f'0{idx}' if idx < 10 else str(idx)

        # 读取图像对
        a_path = os.path.join(self.root, 'Lytro', f'lytro-{idx_str}-A.jpg')
        b_path = os.path.join(self.root, 'Lytro', f'lytro-{idx_str}-B.jpg')

        a = Image.open(a_path).convert('RGB')
        b = Image.open(b_path).convert('RGB')

        # 缩小尺寸节省内存
        a = a.resize((256, 256))
        b = b.resize((256, 256))

        # 转为tensor
        a_tensor = self._tensor(a)  # [3,256,256]
        b_tensor = self._tensor(b)  # [3,256,256]

        # ✅ 修复：直接在tensor上计算灰度图
        # 方法：取RGB三个通道的平均值
        a_gray = a_tensor.mean(dim=0, keepdim=True)  # [1,256,256]
        b_gray = b_tensor.mean(dim=0, keepdim=True)  # [1,256,256]

        # GT用两张灰度图的均值
        gt = (a_gray + b_gray) / 2  # [1,256,256]

        # 数据增强（仅训练时）
        if self.train_transform and self.train:
            # 保持a和b使用相同的随机变换
            seed = torch.randint(0, 1000000, (1,)).item()

            # 注意：PIL图像才能做transform
            a_pil = transforms.ToPILImage()(a_tensor)
            b_pil = transforms.ToPILImage()(b_tensor)

            torch.manual_seed(seed)
            a_pil = self.train_transform(a_pil)

            torch.manual_seed(seed)
            b_pil = self.train_transform(b_pil)

            # 重新转tensor
            a = self._tensor(a_pil)
            b = self._tensor(b_pil)
        else:
            a = a_tensor
            b = b_tensor

        return a, b, gt


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)