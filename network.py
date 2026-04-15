# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:05:21 2020
Modified: Added CBAM attention mechanism
"""
from blocks import ResBlock, Conv2dBlock, CBAM
import torch.nn as nn
import torch
from kornia.color import RgbToGrayscale


class Net(nn.Module):
    def __init__(self, num_blocks, num_feat, use_cbam=False, reduction=16):
        super(Net, self).__init__()
        self.use_cbam = use_cbam

        # 初始卷积层
        self.conv0 = Conv2dBlock(1, num_feat, 3)

        # 构建主网络
        main_layers = []
        for i in range(num_blocks):
            main_layers.append(ResBlock(num_feat, 3))
            # 在每个残差块后添加CBAM
            if use_cbam:
                main_layers.append(CBAM(num_feat, reduction))

        self.main = nn.Sequential(*main_layers)

        # 融合卷积层
        self.conv1 = Conv2dBlock(num_feat * 2, 1, 3, act=None, norm=None)

        print(f'[Network] Using CBAM: {use_cbam}')

    def forward(self, a, b):
        a = RgbToGrayscale()(a)
        b = RgbToGrayscale()(b)

        a = self.main(self.conv0(a))
        b = self.main(self.conv0(b))

        return torch.sigmoid(self.conv1(torch.cat((a, b), dim=1)))


# 为了方便使用，定义两个类
class NetOriginal(Net):
    def __init__(self, num_blocks, num_feat):
        super(NetOriginal, self).__init__(num_blocks, num_feat, use_cbam=False)


class NetWithCBAM(Net):
    def __init__(self, num_blocks, num_feat, reduction=16):
        super(NetWithCBAM, self).__init__(num_blocks, num_feat, use_cbam=True, reduction=reduction)