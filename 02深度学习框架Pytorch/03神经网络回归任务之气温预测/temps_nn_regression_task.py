# 用于数值计算
import numpy as np
# 用于数据处理与分析
import pandas as pd
# 用于绘制图表
import matplotlib.pyplot as plt
# PyTorch深度学习框架
import torch
# PyTorch优化器模块
import torch.optim as optim
# 用于警告控制
import warnings
# 用于处理日期时间
import datetime
# 用于数据预处理
from sklearn import preprocessing

# 忽略代码运行中可能出现的警告信息
warnings.filterwarnings("ignore")

# 使用pandas读取名为'temps.csv'的CSV文件，将历史气温数据存储在features变量中
features = pd.read_csv('data/temps.csv')
# 查看数据
features.head()
"""
    数据集中各个字段的含义
        1、year,moth,day,week分别表示的具体的时间
        2、temp_2：前天的最高温度值
        3、temp_1：昨天的最高温度值
        4、average：在历史中，每年这一天的平均最高温度值
        5、actual：这就是我们的标签值了，当天的真实最高温度
        6、friend：这一列可能是凑热闹的，你的朋友猜测的可能值，忽略即可
"""

# 打印数据的维度信息，显示数据集有多少行和列。
print('数据维度:', features.shape)

# 从数据集中分别提取年、月、日三列数据。
years = features['year']
months = features['month']
days = features['day']

# 将年月日转换为datetime格式
# 首先将年月日拼接成'YYYY-MM-DD'格式的字符串
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# 使用datetime.datetime.strptime()将字符串转换为datetime对象
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 准备画图
# 指定默认风格，设置matplotlib图表的样式为'fivethirtyeight'风格，一种美观的预设样式
plt.style.use('fivethirtyeight')

# 设置布局，创建一个2x2的子图布局，整体图像大小为10x10英寸
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
# fig.autofmt_xdate(rotation = 45)用于自动格式化日期标签并将其旋转45度，以便更好地显示。
fig.autofmt_xdate(rotation = 45)

# 标签值
ax1.plot(dates, features['actual'])
# 在第一个子图(ax1)中绘制实际温度值随时间的变化曲线，设置y轴标签为'Temperature'，标题为'Max Temp'
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# 昨天列
ax2.plot(dates, features['temp_1'])
# 在第二个子图(ax2)中绘制昨天温度值随时间的变化曲线，设置相应标签和标题
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# 前天列
ax3.plot(dates, features['temp_2'])
# 在第三个子图(ax3)中绘制前天温度值随时间变化的曲线，设置相应标签和标题
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

# friends列
ax4.plot(dates, features['friend'])
# 在第四个子图(ax4)中绘制朋友猜测温度值随时间变化的曲线，设置相应标签和标题
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

# 自动调整子图参数，使子图之间有适当的间距，避免重叠。
plt.tight_layout(pad=2)

# 独热编码
# 对特征数据进行独热编码(one-hot encoding)处理，将分类变量转换为二进制向量
features = pd.get_dummies(features)
# features.head(5)显示处理后的前5行数据
features.head(5)

# 将实际温度值('actual'列)提取出来作为标签，并转换为numpy数组。
labels = np.array(features['actual'])

# 从特征数据中删除'actual'列，因为这是我们要预测的目标值，不应该作为特征输入到模型中
features= features.drop('actual', axis = 1)

# 将处理后的特征列名保存到feature_list中，以备后续使用
feature_list = list(features.columns)

# 将特征数据转换为numpy数组格式，便于后续处理。
features = np.array(features)
# 查看特征数据的形状(维度)
features.shape

# 使用sklearn的StandardScaler对特征数据进行标准化处理，使数据均值为0，标准差为1
input_features = preprocessing.StandardScaler().fit_transform(features)
# 显示第一条数据标准化后的结果
input_features[0]

# 将标准化后的特征数据和标签数据转换为PyTorch张量(tensor)格式，便于在PyTorch中进行计算。
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# 手动初始化神经网络的权重和偏置参数，所有参数都设置为需要计算梯度(requires_grad=True)，以便进行反向传播。
# weights: 输入层到隐藏层的权重(14个输入特征，128个隐藏单元)
weights = torch.randn((14, 128), dtype=float, requires_grad=True)
# biases: 隐藏层的偏置(128个)
biases = torch.randn(128, dtype=float, requires_grad=True)
# weights2: 隐藏层到输出层的权重(128个隐藏单元，1个输出)
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
# biases2: 输出层的偏置(1个)
biases2 = torch.randn(1, dtype=float, requires_grad=True)


# 设置学习率为0.001
learning_rate = 0.001
# 创建一个空列表用于存储训练过程中的损失值
losses = []

# 开始训练循环，总共进行1000次迭代
for i in range(1000):
    # 计算隐藏层的输出：将输入数据x与权重weights相乘，再加上偏置biases
    hidden = x.mm(weights) + biases
    # 对隐藏层输出应用ReLU激活函数，增加模型的非线性表达能力
    hidden = torch.relu(hidden)
    # 计算输出层的预测结果：将隐藏层输出hidden与权重weights2相乘，再加上偏置biases2
    predictions = hidden.mm(weights2) + biases2
    # 计算损失函数，这里使用均方误差(MSE)作为损失函数，即预测值与真实值差的平方的平均值
    loss = torch.mean((predictions - y) ** 2)
    # 损失值添加到losses列表中用于后续分析
    losses.append(loss.data.numpy())

    # 每隔100次迭代打印一次当前的损失值，用于监控训练过程
    if i % 100 == 0:
        print('loss:', loss)
    # 进行反向传播计算，自动计算损失函数对各个参数的梯度
    loss.backward()

    # 手动更新网络参数，使用梯度下降法
    # 更新输入层到隐藏层的权重
    weights.data.add_(- learning_rate * weights.grad.data)
    # 更新输入层到隐藏层的偏置
    biases.data.add_(- learning_rate * biases.grad.data)
    # 更新隐藏层到输出层的权重
    weights2.data.add_(- learning_rate * weights2.grad.data)
    # 更新隐藏层到输出层的偏置
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 在每次迭代结束后，将所有参数的梯度清零，为下一次迭代做准备。这是因为在PyTorch中，梯度是累积的，如果不手动清零，梯度会不断累加。
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

# 用于查看预测结果的形状
predictions.shape


"""

"""

# 输入特征的数量，等于特征数据的列数
input_size = input_features.shape[1]
# 隐藏层神经元数量
hidden_size = 128
# 输出层神经元数量（预测一个值）
output_size = 1
# 批处理大小
batch_size = 16
# 使用PyTorch的Sequential容器构建神经网络
my_nn = torch.nn.Sequential(
    # 第一层：线性层，将input_size维输入映射到hidden_size维
    torch.nn.Linear(input_size, hidden_size),
    # 激活函数：Sigmoid函数（与之前手动实现的ReLU不同）
    torch.nn.Sigmoid(),
    # 第二层：线性层，将hidden_size维输入映射到output_size维
    torch.nn.Linear(hidden_size, output_size),
)
# 定义损失函数和优化器
# 均方误差损失函数（MSE）
cost = torch.nn.MSELoss(reduction='mean')
# Adam优化器，学习率0.001，优化my_nn的所有参数
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)


# 创建一个空列表用于存储训练过程中的损失值。
losses = []
# 开始训练循环，共1000次迭代。在每次迭代中使用小批量(mini-batch)梯度下降
for i in range(1000):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(input_features), batch_size):
        # 内层循环以batch_size为步长遍历所有数据
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        # start和end确定当前批次的数据范围
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        # 将当前批次的数据转换为PyTorch张量
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        # 前向传播并计算损失
        # prediction：通过网络得到预测值
        prediction = my_nn(xx)
        # loss：计算预测值与真实值之间的损失
        loss = cost(prediction, yy)
        # optimizer.zero_grad()：清零优化器中的梯度
        optimizer.zero_grad()
        # loss.backward()：计算梯度，retain_graph=True保留计算图
        loss.backward(retain_graph=True)
        # optimizer.step()：更新网络参数
        optimizer.step()
        # 将当前批次的损失值添加到batch_loss列表中
        batch_loss.append(loss.data.numpy())

    # 每隔100次迭代，计算并打印当前批次的平均损失。
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

    # 使用整个数据集进行预测，并将结果转换为numpy数组
    x = torch.tensor(input_features, dtype=torch.float)
    predict = my_nn(x).data.numpy()

    # 重新处理日期格式
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # 创建一个表格来存日期和其对应的标签数值
    true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

    # 同理，从特征数据中提取月份、日期和年份信息
    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]

    # 重新构建日期格式
    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

    # 创建包含预测结果的DataFrame
    # date列：包含日期信息
    # prediction列：包含模型预测的温度值，
    # predict.reshape(-1)将预测结果展平为一维数组
    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

    # 绘制真实温度值的曲线图：
    # x轴：日期
    # y轴：实际温度值
    # ['b-']表示蓝色实线
    # [label='actual']设置图例标签为'actual'
    plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

    # 绘制预测温度值的散点图：
    # x轴：日期
    # y轴：预测温度值
    # ['ro']表示红色圆点
    # [label='prediction']设置图例标签为'prediction'
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
    # 设置图表属性：
    # plt.xticks(rotation='60')：将x轴刻度标签旋转60度，便于阅读
    plt.xticks(rotation='60');
    # plt.legend()：显示图例，区分真实值和预测值
    plt.legend()

    # 设置图表的标签和标题：
    # plt.xlabel('Date')：设置x轴标签为'Date'
    plt.xlabel('Date');
    # plt.ylabel('Maximum Temperature (F)')：设置y轴标签为'Maximum Temperature (F)'
    plt.ylabel('Maximum Temperature (F)');
    # plt.title('Actual and Predicted Values')：设置图表标题为'Actual and Predicted Values'
    plt.title('Actual and Predicted Values');
