# 导入pytorch核心库
import torch
# 导入pytorch神经网络模块
import torch.nn as nn
# 导入pytorch的优化器模块
import torch.optim as optim
# 导入pytorch的函数式接口
import torch.nn.functional as F
# 导入pytorch的视觉数据集和转换工具
from torchvision import datasets,transforms
# 导入matplotlib数据可视化处理
import matplotlib.pyplot as plt
# 导入numpy用于数据可视化处理
import numpy as np


"""
    读取数据
        1、分别构建训练集和测试集(验证集)
        2、使用DataLoader迭代取数据
"""

# 输入图像的大小为28*28
input_size = 28
# 标签的种类数
num_classes = 10
# 训练的循环周期
num_epochs = 3
# 批次的大小，64张图片
batch_size = 64

# 加载MNIST数据集
# 训练集设置train=True,使用ToTensor()转换将图像转换为Tensor,并设置download=True自动下载数据集
train_dataset = datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 测试集设置train=False,使用ToTensor()转换将图像转换为Tensor
test_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# 创建数据加载器。使用DataLoader将数据集包装成批次，设置批次大小为64，shuffle=True表示每个epoch打乱数据顺序
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

"""
    卷积网络构建
        1、一般卷积层，relu层，池化层可以打包一起
        2、需要注意的是，卷积最后结果还是一个特征图，需要把图转换成向量才能做分类或者回归任务
"""


class CNN(nn.Module):
    def __init__(self):
        # init方法初始化网络结构，调用父类构造函数
        super(CNN, self).__init__()
        # 定义第一个卷积层。
        self.conv1 = nn.Sequential(  # 输入大小 (1, 28, 28)
            nn.Conv2d(
                # 输入为1通道（灰度图）
                in_channels=1,
                # 输出16个特征图
                out_channels=16,  # 要得到几多少个特征图
                # 卷积核大小为5×5
                kernel_size=5,
                # 步长为1
                stride=1,
                # 填充为2（保持尺寸不变），如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2
                padding=2,  #
            ),  # 输出的特征图为 (16, 28, 28)
            # relu层
            nn.ReLU(),
            # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )
        # 定义第二个卷积层。
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            # 输入16通道，输出32通道，卷积核5×5，步长1，填充2
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            # ReLU激活函数
            nn.ReLU(),
            # 卷积层，输入32通道，输出32通道
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            # 2×2的最大池化层，输出尺寸变为(32, 7, 7)
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        # 定义第三个卷积层。
        self.conv3 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            # 输入32通道，输出64通道，卷积核5×5，步长1，填充2
            nn.Conv2d(32, 64, 5, 1, 2),  # 输出 (32, 14, 14)
            # ReLU激活函数
            nn.ReLU(),  # 输出 (32, 7, 7)
        )
        # 定义输出层。将卷积层输出展平后连接到一个全连接层，输入维度为64×7×7，输出维度为10（对应10个类别）。
        self.out = nn.Linear(64 * 7 * 7, 10)

    # 定义前向传播过程,数据依次通过三个卷积层，然后，最后
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 通过view函数展平为一维向量 结果为：(batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        # 通过全连接层输出结果
        out_put = self.out(x)
        return out_put

"""
    准确率评估
"""

# 定义准确率计算函数
def accuracy(predictions, labels):
    # 使用torch.max获取预测结果中概率最大的类别
    prediction_category = torch.max(predictions.data, 1)[1]
    # 与真实标签比较，计算正确预测数量
    rights = prediction_category.eq(labels.data.view_as(prediction_category)).sum()
    return rights, len(labels)


# 创建CNN网络实例
net = CNN()
# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()
# 优化器使用Adam，学习率设置为0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)


"""
    训练网络模型
"""

# 开始训练循环,共3个epoch
for epoch in range(num_epochs):
    # 当前epoch的结果保存下来
    train_rights = []

    # 针对容器中的每一个批进行循环
    for batch_idx, (data, target) in enumerate(train_loader):
        # 使用net.train()设置网络为训练模式
        net.train()
        # 前向传播得到输出
        output = net(data)
        # 计算损失
        loss = criterion(output, target)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 参数更新
        optimizer.step()
        # 准确率计算并保存
        right = accuracy(output, target)
        train_rights.append(right)

        # 每100个批次在测试集上验证一次模型性能
        if batch_idx % 100 == 0:
            # 使用net.eval()设置网络为评估模式
            net.eval()
            val_rights = []

            # 遍历测试集
            for (data, target) in test_loader:
                # 验证集上前向传播得到输出
                output = net(data)
                # 在测试集上计算准确率
                right = accuracy(output, target)
                val_rights.append(right)

            # 计算训练集和测试集的总体准确率
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            # 打印当前训练状态，包括epoch、进度、损失值、训练集准确率和测试集准确率
            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))