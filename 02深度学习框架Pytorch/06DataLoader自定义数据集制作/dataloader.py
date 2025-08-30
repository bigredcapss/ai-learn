#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集Dataloader制作

如何自定义数据集：
1. 数据和标签的目录结构先搞定(得知道到哪读数据)
2. 写好读取数据和标签路径的函数(根据自己数据集情况来写)
3. 完成单个数据与标签读取函数(给dataloader举一个例子)

以花朵数据集为例：
- 原来数据集都是以文件夹为类别ID，现在咱们换一个套路，用txt文件指定数据路径与标签(实际情况基本都这样)
- 这回咱们的任务就是在txt文件中获取图像路径与标签，然后把他们交给dataloader
- 核心代码非常简单，按照对应格式传递需要的数据和标签就可以啦
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 忽略警告
warnings.filterwarnings('ignore')

def load_annotations(ann_file):
    """
    读取txt文件中的路径和标签
    第一个小任务，从标注文件中读取数据和标签
    至于你准备存成什么格式，都可以的，一会能取出来东西就行
    """
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for filename, gt_label in samples:
            data_infos[filename] = np.array(gt_label, dtype=np.int64)
    return data_infos

def test_load_annotations():
    """测试读取标注文件"""
    print("测试读取标注文件:")
    print(load_annotations('./flower_data/train.txt'))

def prepare_data_lists():
    """
    分别把数据和标签都存在list里
    不是我非让你存list里，因为dataloader到时候会在这里取数据
    按照人家要求来，不要耍个性，让整list咱就给人家整
    """
    img_label = load_annotations('./flower_data/train.txt')
    image_name = list(img_label.keys())
    label = list(img_label.values())
    return image_name, label

def prepare_image_paths():
    """
    图像数据路径得完整
    因为一会咱得用这个路径去读数据，所以路径得加上前缀
    以后大家任务不同，数据不同，怎么加你看着来就行，反正得能读到图像
    """
    data_dir = './flower_data/'
    train_dir = data_dir + '/train_filelist'
    valid_dir = data_dir + '/val_filelist'
    
    image_name, _ = prepare_data_lists()
    image_path = [os.path.join(train_dir, img) for img in image_name]
    return train_dir, valid_dir, image_path

class FlowerDataset(Dataset):
    """
    自定义数据集类
    把上面那几个事得写在一起
    1.注意要使用from torch.utils.data import Dataset, DataLoader
    2.类名定义class FlowerDataset(Dataset)，其中FlowerDataset可以改成自己的名字
    3.def __init__(self, root_dir, ann_file, transform=None):咱们要根据自己任务重写
    4.def __getitem__(self, idx):根据自己任务，返回图像数据和标签数据
    """
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform
 
    def __len__(self):
        return len(self.img)
 
    def __getitem__(self, idx):
        image = Image.open(self.img[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label
    
    def load_annotations(self):
        """加载标注文件"""
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos

def get_data_transforms():
    """
    数据预处理(transform)
    1.预处理的事都在上面的__getitem__中完成，需要对图像和标签咋咋地的，要整啥事，都在上面整
    2.返回的数据和标签就是建模时模型的输入和损失函数中标签的输入，一定整明白自己模型要啥
    3.预处理这个事是你定的，不同的数据需要的方法也不一样，下面给出的是比较通用的方法
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(64),
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(64),  # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_dataloaders():
    """
    根据写好的class FlowerDataset(Dataset):来实例化咱们的dataloader
    1.构建数据集：分别创建训练和验证用的数据集（如果需要测试集也一样的方法）
    2.用Torch给的DataLoader方法来实例化(batch啥的自己定，根据你的显存来选合适的)
    3.打印看看数据里面是不是有东西了
    """
    train_dir, valid_dir, _ = prepare_image_paths()
    data_transforms = get_data_transforms()
    
    train_dataset = FlowerDataset(root_dir=train_dir, ann_file='./flower_data/train.txt', transform=data_transforms['train'])
    val_dataset = FlowerDataset(root_dir=valid_dir, ann_file='./flower_data/val.txt', transform=data_transforms['valid'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader

def test_dataloader():
    """
    用之前先试试，整个数据和标签对应下，看看对不对
    1.别着急往模型里传，对不对都不知道呢
    2.用这个方法：iter(train_loader).next()来试试，得到的数据和标签是啥
    3.看不出来就把图画出来，标签打印出来，确保自己整的数据集没啥问题
    """
    train_loader, val_loader = create_dataloaders()
    
    # 测试训练集
    image, label = iter(train_loader).next()
    sample = image[0].squeeze()
    sample = sample.permute((1, 2, 0)).numpy()
    sample *= [0.229, 0.224, 0.225]
    sample += [0.485, 0.456, 0.406]
    plt.figure(figsize=(6, 6))
    plt.imshow(sample)
    plt.title(f'训练集样本 - 标签: {label[0].numpy()}')
    plt.show()
    print(f'训练集标签是: {label[0].numpy()}')
    
    # 测试验证集
    image, label = iter(val_loader).next()
    sample = image[0].squeeze()
    sample = sample.permute((1, 2, 0)).numpy()
    sample *= [0.229, 0.224, 0.225]
    sample += [0.485, 0.456, 0.406]
    plt.figure(figsize=(6, 6))
    plt.imshow(sample)
    plt.title(f'验证集样本 - 标签: {label[0].numpy()}')
    plt.show()
    print(f'验证集标签是: {label[0].numpy()}')

def setup_model():
    """设置模型"""
    model_name = 'resnet'  # 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    feature_extract = True
    
    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()
    
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102))
    input_size = 64
    
    return model_ft, device

def setup_training():
    """设置训练参数"""
    model_ft, device = setup_model()
    
    # 优化器设置
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
    criterion = nn.CrossEntropyLoss()
    
    return model_ft, optimizer_ft, scheduler, criterion, device

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False, filename='best.pth'):
    """
    训练模型
    咋用就是你来定了，把模型啥的整好往里面传吧
    下面这些事之前都唠过了，按照自己习惯的方法整就得了
    """
    since = time.time()
    best_acc = 0
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()   # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),  # 优化器的状态信息
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step()  # 学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

def main():
    """主函数"""
    print("开始数据集Dataloader制作...")
    
    # 测试读取标注文件
    test_load_annotations()
    
    # 测试数据加载器
    print("\n测试数据加载器...")
    test_dataloader()
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader = create_dataloaders()
    dataloaders = {'train': train_loader, 'valid': val_loader}
    
    # 设置模型和训练参数
    print("\n设置模型和训练参数...")
    model_ft, optimizer_ft, scheduler, criterion, device = setup_training()
    
    # 开始训练
    print("\n开始训练模型...")
    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(
        model_ft, dataloaders, criterion, optimizer_ft, device, num_epochs=20, filename='best.pth'
    )
    
    print("训练完成！")

if __name__ == "__main__":
    main()
