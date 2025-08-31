# PyTorch自定义数据集制作项目说明书

## 📋 项目概述

本项目是一个基于PyTorch框架的自定义数据集制作工具，专门用于处理图像分类任务。项目以花朵数据集为例，演示了如何从原始数据组织到最终模型训练的完整流程，包括数据预处理、自定义数据集类创建、DataLoader配置以及模型训练等核心功能。

### 🎯 项目目标
- 掌握PyTorch自定义数据集的制作方法
- 学习数据预处理和增强技术
- 理解DataLoader的工作原理和使用方法
- 实现完整的深度学习训练流程

## 🏗️ 项目结构

```
06DataLoader自定义数据集制作/
├── dataloader.py              # 主要的Python脚本文件
├── dataloader.ipynb           # Jupyter notebook版本
├── flower_data/               # 数据集目录
│   ├── train/                 # 原始训练数据（按类别分文件夹）
│   ├── valid/                 # 原始验证数据（按类别分文件夹）
│   ├── train_filelist/        # 处理后的训练图像文件
│   ├── val_filelist/          # 处理后的验证图像文件
│   ├── train.txt              # 训练集标注文件（图像名 标签）
│   ├── val.txt                # 验证集标注文件（图像名 标签）
│   └── filelist.py            # 数据预处理脚本
└── 项目说明书.md              # 本文档
```

## 🔧 功能特性

### 1. 数据预处理功能
- **自动生成标注文件**：将按文件夹分类的图像数据转换为txt格式的标注文件
- **文件路径管理**：自动整理图像文件路径，便于DataLoader读取
- **数据格式转换**：支持从文件夹结构到文件列表的转换

### 2. 自定义数据集类
- **FlowerDataset类**：继承PyTorch Dataset类，实现自定义数据集
- **灵活的数据加载**：支持txt格式的标注文件读取
- **数据变换支持**：集成PyTorch transforms，支持数据增强

### 3. 数据增强技术
- **几何变换**：随机旋转、水平/垂直翻转、中心裁剪
- **颜色变换**：亮度、对比度、饱和度、色相调整
- **灰度转换**：随机灰度化处理
- **标准化处理**：ImageNet预训练模型的标准化参数

### 4. 模型训练功能
- **ResNet18模型**：使用预训练的ResNet18作为基础模型
- **迁移学习**：修改最后一层适配102类花朵分类
- **训练监控**：实时显示训练损失、准确率等指标
- **模型保存**：自动保存最佳模型权重

## 🛠️ 技术架构

### 核心技术栈
- **PyTorch**: 深度学习框架
- **torchvision**: 计算机视觉工具包
- **PIL/Pillow**: 图像处理库
- **matplotlib**: 数据可视化
- **numpy**: 数值计算库

### 核心模块设计

#### 1. 数据加载模块
```python
def load_annotations(ann_file):
    """读取txt文件中的图像路径和标签"""
    
class FlowerDataset(Dataset):
    """自定义数据集类"""
```

#### 2. 数据预处理模块
```python
def get_data_transforms():
    """定义训练和验证的数据变换"""
```

#### 3. 模型训练模块
```python
def train_model(model, dataloaders, criterion, optimizer, device):
    """完整的训练循环"""
```

## 📖 使用说明

### 环境要求
```bash
# 必需的Python包
torch>=1.7.0
torchvision>=0.8.0
PIL>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
```

### 安装步骤
1. 克隆或下载项目文件
2. 安装依赖包：
   ```bash
   pip install torch torchvision pillow matplotlib numpy
   ```
3. 准备数据集（按文件夹分类的图像数据）

### 数据准备
1. 将图像数据按类别分文件夹存放：
   ```
   train/
   ├── 1/          # 类别1的图像
   ├── 2/          # 类别2的图像
   └── ...
   
   valid/
   ├── 1/          # 类别1的图像
   ├── 2/          # 类别2的图像
   └── ...
   ```

2. 运行数据预处理脚本：
   ```bash
   cd flower_data
   python filelist.py
   ```

### 运行项目
```bash
# 直接运行Python脚本
python dataloader.py
```

### 主要功能演示

#### 1. 测试数据加载
```python
# 测试读取标注文件
test_load_annotations()

# 测试数据加载器
test_dataloader()
```

#### 2. 创建数据加载器
```python
# 创建训练和验证数据加载器
train_loader, val_loader = create_dataloaders()
```

#### 3. 模型训练
```python
# 设置模型和训练参数
model_ft, optimizer_ft, scheduler, criterion, device = setup_training()

# 开始训练
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
    model_ft, dataloaders, criterion, optimizer_ft, device, num_epochs=20, filename='best.pth'
)
```

## 📊 数据集信息

### 花朵数据集
- **类别数量**: 102类
- **训练集大小**: 约6,553张图像
- **验证集大小**: 约819张图像
- **图像格式**: JPG
- **标注格式**: txt文件（图像名 标签）

### 数据标注格式
```
image_06734.jpg 0
image_06735.jpg 0
image_07086.jpg 9
...
```

## 🔍 核心算法

### 1. 自定义数据集实现
```python
class FlowerDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        # 初始化数据集
        
    def __len__(self):
        # 返回数据集大小
        
    def __getitem__(self, idx):
        # 返回单个样本和标签
```

### 2. 数据增强策略
- **训练时增强**：随机旋转、翻转、颜色抖动、灰度化
- **验证时增强**：仅进行标准化处理
- **标准化参数**：ImageNet预训练模型的均值和标准差

### 3. 训练策略
- **优化器**: Adam优化器
- **学习率**: 初始学习率1e-3，每7个epoch衰减为原来的1/10
- **损失函数**: CrossEntropyLoss
- **批次大小**: 64
- **训练轮数**: 20个epoch

## 📈 性能指标

### 训练监控指标
- **训练损失** (Train Loss)
- **验证损失** (Valid Loss)
- **训练准确率** (Train Accuracy)
- **验证准确率** (Valid Accuracy)
- **学习率变化** (Learning Rate)

### 模型保存策略
- 自动保存验证集上表现最好的模型
- 保存内容包括：模型权重、最佳准确率、优化器状态

## 🚀 扩展功能

### 1. 支持其他数据集
项目设计具有通用性，可以轻松适配其他图像分类数据集：
- 修改类别数量
- 调整数据路径
- 自定义数据增强策略

### 2. 模型选择
支持多种预训练模型：
- ResNet系列 (18, 34, 50, 101, 152)
- AlexNet
- VGG系列
- DenseNet
- Inception

### 3. 训练策略优化
- 支持学习率调度器
- 支持早停机制
- 支持模型集成

## 🐛 常见问题

### Q1: 如何处理不同格式的图像文件？
A: PIL库支持多种图像格式，包括JPG、PNG、BMP等，无需额外处理。

### Q2: 如何调整批次大小？
A: 在`create_dataloaders()`函数中修改`batch_size`参数。

### Q3: 如何添加新的数据增强方法？
A: 在`get_data_transforms()`函数中添加新的transform操作。

### Q4: 如何保存训练过程中的损失曲线？
A: 训练函数已返回损失历史，可以用于绘制损失曲线。

## 📝 开发日志

### 版本历史
- **v1.0**: 基础功能实现，支持花朵数据集训练
- **v1.1**: 添加数据可视化功能
- **v1.2**: 优化训练流程，添加模型保存功能

### 待优化功能
- [ ] 添加TensorBoard支持
- [ ] 实现多GPU训练
- [ ] 添加数据增强可视化
- [ ] 支持更多数据集格式
