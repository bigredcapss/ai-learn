# 图像识别模型训练与测试说明文档

## 项目概述

本项目是一个基于PyTorch的图像识别系统，主要用于花卉分类任务。系统使用预训练的ResNet-18模型作为特征提取器，通过迁移学习的方式对102种不同花卉进行分类识别。

## 功能特性

1. **数据预处理**：
   - 对训练数据进行多种数据增强操作（旋转、裁剪、翻转、颜色抖动等）
   - 对验证数据进行标准化处理
   - 支持批量数据加载

2. **模型训练**：
   - 使用预训练的ResNet-18模型进行迁移学习
   - 支持冻结预训练层参数，仅训练分类层
   - 实现完整的训练循环和验证过程
   - 自动保存最佳模型权重

3. **模型评估**：
   - 加载训练好的模型进行预测
   - 可视化预测结果
   - 显示预测准确率

## 依赖库

- torch
- torchvision
- matplotlib
- numpy
- json
- PIL
- copy
- time
- os

## 数据集结构

项目需要以下数据集结构：

```
flower_data/
├── train/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── ...
└── valid/
    ├── 1/
    ├── 2/
    ├── 3/
    └── ...
```

其中，每个数字文件夹代表一种花卉类别。

## 代码结构说明

### 1. 数据准备阶段

```python
data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
```

定义了数据目录路径，包括训练数据和验证数据。

### 2. 数据预处理

```python
data_transforms = {
    'train': transforms.Compose([...]),
    'valid': transforms.Compose([...])
}
```

- 训练数据变换包括：调整大小、随机旋转、中心裁剪、随机水平翻转、随机垂直翻转、颜色抖动、随机灰度化、转换为张量和标准化
- 验证数据变换包括：调整大小、转换为张量和标准化

### 3. 数据加载器

```python
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
```

使用ImageFolder创建训练和验证数据集，并创建对应的数据加载器。

### 4. 模型初始化

```python
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 102)
    input_size = 64
    return model_ft, input_size
```

使用预训练的ResNet-18模型，替换最后的全连接层以适应102个花卉类别。

### 5. 训练过程

```python
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt')
```

训练函数实现了完整的训练循环：
- 支持训练和验证两个阶段
- 记录训练过程中的损失和准确率
- 保存最佳模型权重
- 使用学习率调度器调整学习率

### 6. 模型评估与可视化

代码最后部分加载训练好的模型，对验证集数据进行预测，并使用matplotlib可视化预测结果。

## 参数配置

- **批处理大小**: 128
- **优化器**: Adam，学习率0.01
- **损失函数**: 交叉熵损失
- **学习率调度**: 每10步将学习率衰减为原来的0.1
- **训练轮数**: 20轮
- **特征提取**: 仅训练最后的分类层（冻结预训练模型参数）

## 运行说明

1. 确保已安装所有依赖库
2. 准备好符合要求格式的数据集
3. 确保cat_to_name.json文件存在（包含类别ID到名称的映射）
4. 运行脚本：`python test_nn.py`

## 输出结果

1. 训练过程中会输出每轮的训练和验证损失及准确率
2. 最佳模型将保存为`best.pt`文件
3. 训练完成后会显示预测结果的可视化图表，绿色标题表示预测正确，红色表示预测错误

## 注意事项

1. 项目默认使用GPU进行训练，如果无GPU则自动切换到CPU
2. 数据集路径需要根据实际环境进行调整
3. 可根据需要调整训练轮数、学习率等超参数