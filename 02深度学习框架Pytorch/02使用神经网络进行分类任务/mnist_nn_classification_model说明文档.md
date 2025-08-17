# mnist_nn_classification_model.py 说明文档

该文件实现了一个基于 PyTorch 的 MNIST 手写数字识别神经网络分类模型。通过下载 MNIST 数据集，构建多层神经网络，并进行训练和验证。

## 1. 功能概述

本代码实现了一个完整的深度学习项目流程，包括：

- MNIST 数据集的下载和预处理
- 多层神经网络模型的定义
- 模型训练和验证过程
- 模型性能评估

## 2. 依赖库

- `torch`: PyTorch 深度学习框架
- `pathlib`: 文件路径处理
- `requests`: HTTP 请求发送
- `pickle`: Python 对象序列化
- `gzip`: 文件解压缩
- `matplotlib`: 数据可视化
- `numpy`: 数值计算
- `torch.nn`: PyTorch 神经网络模块
- `torch.nn.functional`: 神经网络函数
- `torch.utils.data`: 数据加载工具
- `torch.optim`: 优化算法

## 3. 代码结构详解

### 3.1 数据准备

```python
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
```

该部分创建数据存储目录并从指定 URL 下载 MNIST 数据集（如果本地不存在）。

### 3.2 数据加载和可视化

```python
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
```

使用 gzip 解压并加载 MNIST 数据集，通过 matplotlib 显示第一个训练样本。

### 3.3 数据转换

```python
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
```

将 NumPy 数组转换为 PyTorch 张量，便于后续神经网络处理。

### 3.4 模型定义

```python
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
```

定义一个三层神经网络模型：

- 输入层到第一个隐藏层：784 → 128 神经元
- 第一个隐藏层到第二个隐藏层：128 → 256 神经元
- 第二个隐藏层到输出层：256 → 10 神经元（对应 10 个数字类别）
- 使用 ReLU 激活函数

### 3.5 数据加载器

```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
```

使用 TensorDataset 和 DataLoader 创建训练和验证数据加载器，支持批量处理和数据打乱。

### 3.6 训练流程

代码定义了三个核心函数：

1. `get_model()`: 创建模型实例和优化器
2. `loss_batch()`: 处理单个批次的损失计算和参数更新
3. `fit()`: 执行完整的训练循环

使用交叉熵损失函数和 SGD 优化器进行训练。

## 4. 训练参数

- 批处理大小：训练集 64，验证集 128
- 学习率：0.001
- 训练轮数：25
- 损失函数：交叉熵损失
- 优化器：随机梯度下降(SGD)

## 5. 运行结果

代码会输出以下信息：

- PyTorch 版本
- 训练数据形状
- 初始损失值
- 每轮训练后的验证集损失

## 6. 使用方法

直接运行脚本即可开始训练：

```bash
python mnist_nn_classification_model.py
```

程序会自动下载数据、构建模型、训练并输出训练过程中的验证损失。
