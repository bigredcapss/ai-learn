# 卷积神经网络参数分析说明文档

## 项目概述

本项目使用PyTorch框架实现了一个卷积神经网络(CNN)模型，用于MNIST手写数字识别任务。该项目展示了如何构建、训练和评估一个CNN模型，同时解释了卷积网络中的关键参数和结构。

## 数据集说明

### MNIST数据集
- 包含60,000个训练样本和10,000个测试样本
- 图像尺寸为28x28像素的灰度图
- 标签为0-9的手写数字类别
- 数据集自动下载并转换为PyTorch Tensor格式

## 代码实现详解

### 1. 导入依赖库
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
```

### 2. 超参数设置
- `input_size = 28`: 输入图像尺寸为28x28
- `num_classes = 10`: 分类类别数（0-9共10个数字）
- `num_epochs = 3`: 训练周期数
- `batch_size = 64`: 每个批次包含64张图像

### 3. 数据加载
使用`datasets.MNIST`加载训练集和测试集，并通过`DataLoader`创建数据迭代器：
- 训练集设置[train=True](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/utils.py#L45-L45)
- 测试集设置[train=False](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/utils.py#L45-L45)
- 使用`ToTensor()`转换将图像转换为Tensor
- 设置`shuffle=True`在每个epoch打乱数据

### 4. CNN模型结构

#### CNN类定义
继承自[nn.Module](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/02%E4%BD%BF%E7%94%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1/mnist_nn_classification_model.py#L8-L27)，包含三个卷积层块和一个输出层：

##### 第一个卷积层块 (conv1)
- `Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)`
  - 输入通道: 1 (灰度图)
  - 输出特征图: 16
  - 卷积核大小: 5x5
  - 步长: 1
  - 填充: 2 (保持尺寸不变)
  - 输出尺寸: (16, 28, 28)
- `ReLU()`: 激活函数
- `MaxPool2d(kernel_size=2)`: 2x2最大池化
  - 输出尺寸: (16, 14, 14)

##### 第二个卷积层块 (conv2)
- `Conv2d(16, 32, 5, 1, 2)`: 
  - 输入通道: 16
  - 输出特征图: 32
  - 输出尺寸: (32, 14, 14)
- `ReLU()`: 激活函数
- `Conv2d(32, 32, 5, 1, 2)`:
  - 输入通道: 32
  - 输出特征图: 32
- `ReLU()`: 激活函数
- `MaxPool2d(2)`: 2x2最大池化
  - 输出尺寸: (32, 7, 7)

##### 第三个卷积层块 (conv3)
- `Conv2d(32, 64, 5, 1, 2)`:
  - 输入通道: 32
  - 输出特征图: 64
- `ReLU()`: 激活函数

##### 输出层 (out)
- `Linear(64 * 7 * 7, 10)`: 全连接层
  - 输入维度: 64×7×7 (展平后的特征图)
  - 输出维度: 10 (对应10个类别)

#### 前向传播过程
数据依次通过三个卷积层块，然后通过`view()`函数展平为一维向量，最后通过全连接层输出结果。

### 5. 训练过程

#### 准确率计算函数
```python
def accuracy(predictions, labels)
```
使用`torch.max`获取预测结果中概率最大的类别，然后与真实标签比较，计算正确的预测数量。

#### 模型训练配置
- 模型实例化: `net = CNN()`
- 损失函数: `nn.CrossEntropyLoss()`
- 优化器: `optim.Adam(net.parameters(), lr=0.001)`

#### 训练循环
1. 总共训练3个epoch
2. 每个epoch中:
   - 使用[net.train()](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/models/TextRNN.py#L32-L37)设置网络为训练模式
   - 前向传播得到输出
   - 计算损失
   - 清零梯度
   - 反向传播计算梯度
   - 更新参数
   - 计算当前批次的准确率并保存

3. 每100个批次在测试集上验证一次:
   - 使用[net.eval()](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/models/TextRNN.py#L39-L40)设置网络为评估模式
   - 在测试集上计算准确率
   - 计算训练集和测试集的总体准确率
   - 打印当前训练状态

## 运行环境

### 依赖库
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

### 运行方法
直接运行`conv_nn_param_anly.py`文件即可开始训练：
```bash
python conv_nn_param_anly.py
```

## 关键知识点

### 卷积层参数
1. **in_channels**: 输入通道数
2. **out_channels**: 输出特征图数量
3. **kernel_size**: 卷积核大小
4. **stride**: 步长
5. **padding**: 填充

### 池化层
- 最大池化(MaxPool2d)用于降低特征图的空间维度
- 有助于减少计算量和防止过拟合

### 网络模式切换
- [net.train()](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/models/TextRNN.py#L32-L37): 设置为训练模式，启用dropout、batch normalization等训练时特有的操作
- [net.eval()](file:///D:/Develop/pyproject/ai-learn/02%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6Pytorch/07LSTM%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/models/TextRNN.py#L39-L40): 设置为评估模式，禁用训练时特有的操作

### 准确率计算
通过比较预测类别和真实标签来计算模型准确率，这是评估分类模型性能的重要指标。