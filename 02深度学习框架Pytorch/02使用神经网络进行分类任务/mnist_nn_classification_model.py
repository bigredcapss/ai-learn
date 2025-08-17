# 导入深度学习框架pytorch库
import torch
# 导入pathlib模块，用于处理文件路径
from pathlib import Path
# 导入 requests 库，用于发送 HTTP 请求
import requests
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle
# 导入gzip模块，用于解压缩.gz格式的文件
import gzip
# 从matplotlib库导入pyplot模块，用于数据可视化
from matplotlib import pyplot
# 导入numpy库并重命名为np，用于数值计算
import numpy as np
# 导入PyTorch的神经网络函数模块，用于激活函数等操作
import torch.nn.functional as F
# 导入PyTorch的神经网络模块
from torch import nn
# 从torch.utils.data导入TensorDataset，用于包装张量数据集
from torch.utils.data import TensorDataset
# 从torch.utils.data导入DataLoader，用于批量加载数据
from torch.utils.data import DataLoader
# 导入PyTorch的优化器模块
from torch import optim

"""
    01检查并创建数据目录,下载 MNIST 数据集文件（如果本地不存在的话）
"""

# 打印torch的版本
print(torch.__version__)


# 声明训练数据路径
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

# 创建数据路径
# parents=True: 如果父目录不存在也会一并创建
# exist_ok=True: 如果目录已存在不会抛出异常
PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

# if not (PATH / FILENAME).exists(): 检查目标文件是否已存在
# content = requests.get(URL + FILENAME).content: 如果文件不存在，发送GET请求下载文件内容
# (PATH / FILENAME).open("wb").write(content): 以二进制写入模式打开文件并写入下载的内容
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


"""
    02加载MNIST数据集并显示第一个训练样本
"""

# (PATH / FILENAME).as_posix()将Path对象转换为字符串路径
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    # 使用pickle.load从打开的文件中加载数据
    # MNIST数据集被组织为三个元组：训练集、验证集和测试集
    # _表示忽略第三个元素（测试集）
    # encoding="latin-1"指定解码方式，用于兼容Python 3读取Python 2保存的pickle文件
    # x_train, y_train: 训练数据和标签
    # x_valid, y_valid: 验证数据和标签
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# x_train[0]获取训练集中的第一个样本
# .reshape((28, 28))将一维数组重塑为28x28的二维图像
# pyplot.imshow显示图像
# cmap="gray"使用灰度色彩映射显示图像
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# 打印训练数据的形状，显示数据集的大小
print(x_train.shape)


"""
    03数据转换为PyTorch张量
"""


# 使用map函数将numpy数组转换为PyTorch张量
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
# n, c = x_train.shape: 获取训练数据的形状，n为样本数，c为特征数(784)
n, c = x_train.shape
# x_train, x_train.shape, y_train.min(), y_train.max(): 这行代码计算但未赋值给变量，仅用于查看数据信息
x_train, x_train.shape, y_train.min(), y_train.max()
# 打印训练数据和标签
print(x_train, y_train)
# 打印训练数据形状
print(x_train.shape)
# 打印标签的最小值和最大值
print(y_train.min(), y_train.max())


"""
    04损失函数和简单模型定义
"""

# 定义交叉熵损失函数
loss_func = F.cross_entropy

# 定义一个简单的线性模型函数
def model(xb):
    # 返回输入与权重矩阵相乘加上偏置的结果
    return xb.mm(weights) + bias

# 设置批处理大小为64
bs = 64
# 获取第一个训练批次数据
xb = x_train[0:bs]
# 获取第一个训练批次标签
yb = y_train[0:bs]
# 初始化权重矩阵，784个输入特征到10个输出类别
weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True)
bs = 64
# 初始化偏置向量，大小为10
bias = torch.zeros(10, requires_grad=True)
# 计算并打印当前批次的损失值
print(loss_func(model(xb), yb))


"""
    05定义神经网络模型类
"""

class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个隐藏层：784->128
        self.hidden1 = nn.Linear(784, 128)
        # 第二个隐藏层：128->256
        self.hidden2 = nn.Linear(128, 256)
        # 输出层：256->10
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        # 第一层+ReLU激活
        x = F.relu(self.hidden1(x))
        # 第二层+ReLU激活
        x = F.relu(self.hidden2(x))
        # 输出层
        x = self.out(x)
        return x

# 创建神经网络实例
net = Mnist_NN()
# 打印网络结构
print(net)
# 遍历并打印模型中所有参数的名称、值和尺寸
for name, parameter in net.named_parameters():
    print(name, parameter,parameter.size())


"""
    06创建数据集和数据加载器
"""

# 创建训练数据集
train_ds = TensorDataset(x_train, y_train)
# 创建训练数据加载器
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
# 创建验证数据集
valid_ds = TensorDataset(x_valid, y_valid)
# 创建验证数据加载器
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


"""
    07定义获取训练和验证数据加载器的函数
"""
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


"""
    08模型训练函数,实现了完整的训练流程：
        get_model函数：
            1、创建神经网络模型实例
            2、配置随机梯度下降(SGD)优化器
        loss_batch函数：
            1、计算单个批次的损失
            2、如果提供了优化器，则执行反向传播和参数更新
            3、返回损失值和样本数量
        fit函数（在上一段代码中定义）：
            1、执行指定步数的训练循环
            2、在训练阶段使用训练数据训练模型
            3、在验证阶段评估模型性能
            4、打印每个步骤的验证损失
"""
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))


# 定义获取模型函数
def get_model():
    # 创建Mnist_NN神经网络实例
    model = Mnist_NN()
    # 返回模型和SGD优化器（学习率为0.001）
    return model, optim.SGD(model.parameters(), lr=0.001)


# 定义批次损失计算函数
def loss_batch(model, loss_func, xb, yb, opt=None):
    #  计算当前批次的损失值
    loss = loss_func(model(xb), yb)

    # 如果提供了优化器，执行反向传播和优化
    if opt is not None:
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        opt.step()
        # 梯度清零
        opt.zero_grad()
    #  返回损失值（标量）和批次大小
    return loss.item(), len(xb)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, valid_dl)