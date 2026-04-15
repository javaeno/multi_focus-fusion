# -*- coding: utf-8 -*-
import torch
import os
import numpy as np

from network import Net  # 用原始网络
from utils import Dataset, mkdir
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt

# 训练配置
batch_size = 1  # 笔记本内存小，用2
lr = 1e-4
num_epoch = 20  # 先训练20个epoch
num_blocks = 8
num_feat = 128

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建网络
net = Net(num_blocks, num_feat).to(device)
print("网络初始化成功")

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 加载数据
print("加载训练数据...")
trainset = Dataset('test_set', train=True)  # 用Lytro前10张训练
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

print("加载测试数据...")
testset = Dataset('test_set', train=False)  # 用Lytro后10张测试
test_loader = DataLoader(testset, batch_size=1, shuffle=False)

# 保存路径
save_path = datetime.datetime.now().strftime("%m-%d-%H-%M")
mkdir(save_path)
print(f"结果保存到: {save_path}")

# 训练记录
train_losses = []
test_losses = []
best_loss = float('inf')

# 开始训练
print("\n" + "=" * 50)
print("开始训练...")
print("=" * 50)

for epoch in range(num_epoch):
    # 训练阶段
    net.train()
    epoch_loss = 0

    for i, (A, B, GT) in enumerate(train_loader):
        A, B, GT = A.to(device), B.to(device), GT.to(device)

        # 数据增强：随机旋转
        if epoch < 10:  # 前10个epoch用数据增强
            temp_int = int(torch.randint(4, [1]))
            A = torch.rot90(A, temp_int, [-1, -2])
            B = torch.rot90(B, temp_int, [-1, -2])
            GT = torch.rot90(GT, temp_int, [-1, -2])

        # 前向传播
        optimizer.zero_grad()
        output = net(A, B)

        # 计算损失
        loss = torch.nn.BCELoss()(output, GT)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 测试阶段
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for A, B, GT in test_loader:
            A, B, GT = A.to(device), B.to(device), GT.to(device)
            output = net(A, B)
            loss = torch.nn.BCELoss()(output, GT)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"\nEpoch [{epoch + 1}/{num_epoch}] 完成")
    print(f"训练 Loss: {avg_train_loss:.4f}, 测试 Loss: {avg_test_loss:.4f}")

    # 保存最佳模型
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(net.state_dict(), os.path.join(save_path, 'best_net.pth'))
        print(f"✓ 保存最佳模型 (loss: {best_loss:.4f})")

    # 每个epoch保存一次检查点
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'test_loss': avg_test_loss,
    }, os.path.join(save_path, f'checkpoint_epoch_{epoch + 1}.pth'))

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, 'loss_curve.png'))
plt.show()

print("\n" + "=" * 50)
print("训练完成！")
print(f"最佳模型保存在: {save_path}/best_net.pth")
print(f"最终训练 Loss: {train_losses[-1]:.4f}")
print(f"最终测试 Loss: {test_losses[-1]:.4f}")
print("=" * 50)