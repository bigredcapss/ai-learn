## Pytorch框架与其他框架的对比分析

https://cloud.tencent.com/developer/article/2389961

## Anaconda安装与基本使用

### 安装

```json

Anaconda下载地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

Anaconda版本：Anaconda3-2020.07-Windows-x86_64

```

### anaconda基本使用

```shell
# 查看anaconda信息
conda info

# 查看anaconda环境
conda info -e

# 创建python虚拟环境
conda create -n py36 python=3.6

# 创建python虚拟环境并指定基础工具包
conda create -n py36 python=3.6 anaconda

# 进入Anaconda Prompt，激活/切换虚拟环境
conda activate py36

# 删除虚拟环境py36
conda remove -n py36 --all


```

### jupyter notebook基本使用
* jupyter notebook是anaconda下的功能，依赖anaconda自带的python环境
* 当使用jupyter notebook调试代码时，需要在anaconda自带的python环境中安装对应的依赖包
* 想要使用哪个目录，就直接进入对应目录下，输入以下命令

```shell

jupyter notebook

```

### pip基本使用

```shell

# 下载对应的依赖包 指定软件源
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 查看安装了哪些包
pip list

```

## IDE安装

Pycharm,Cursor,Visual Code都可,不限定版本

## Pytorch框架CPU与GPU版本安装

> pytorch官网: https://pytorch.org/get-started/locally/
> 资源下载: https://download.pytorch.org/whl/

### cuda与torch与torchversion版本对应关系

> 参考1：https://blog.csdn.net/shiwanghualuo/article/details/122860521
> 参考2：https://download.pytorch.org/whl/torch_stable.html

### CPU版本安装

适用于所有设备，前提需要安装anaconda

```shell

pip install torch==1.10.1 -i https://download.pytorch.org/whl/torch_stable.html

```

### GPU版本安装

> cuda下载：https://developer.nvidia.com/cuda-toolkit-archive

#### GPU要求

* 需检查显存大小，至少6GB以上;小于等于4GB不建议安装GPU版本；8GB显存以上为佳

##### 安装cuda

> 参考：https://zhuanlan.zhihu.com/p/23071389812
> 参考：https://blog.csdn.net/zyh20041111/article/details/139214911
> 进入https://developer.nvidia.com/cuda-toolkit-archive，下载cuda，建议下载11.3.x版本

```shell
# 查看当前显卡信息
nvidia-smi
# 安装完成后，查看cuda版本
nvcc --version 或者nvcc -V

```

##### 安装pytorch

> 踩坑参考：https://blog.csdn.net/qq_44832009/article/details/129351554

* 建议手动安装，先把cuda对应的pytorch下载下来后，进行手动安装
* 下载地址 https://download.pytorch.org/whl/torch_stable.html
* 使用pytorch提供的版本对应关系资源筛选，下载后，指定下载文件进行安装，可以避免版本不对应问题

```shell
#手动安装前，需检查当前python环境版本，如果和手动下载的torch版本不一致，会安装不成功
可以通过Anaconda Prompt切换到对应的python环境版本，切换到下载torch所在的目录
# 手动安装下载的pytorch
pip install "torch-1.10.2+cu113-cp36-cp36m-win_amd64.whl"

# 测试cuda是否可用，输出True表示成功
import torch

print(torch.cuda.is_available())

```

### 注意事项

* 确保python版本，pytorch版本，pytorchversion版本，cuda版本的一致性
* 根据python版本选择pytorch版本，选择pytorchversion，cuda版本
