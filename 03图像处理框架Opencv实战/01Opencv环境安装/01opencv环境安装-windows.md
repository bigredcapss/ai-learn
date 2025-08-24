## Opencv是什么

https://www.runoob.com/opencv/opencv-intro.html


## 前期准备

```json
需要提前安装anaconda和python环境

anaconda需要使用其中的2个工具：juypter notebook和 Anaconda Prompt

```


## Opencv安装

### 通过pip安装opencv

#### pip安装

```shell
# opencv-python==3.4.9.33需要python版本大于3.6
pip install opencv-python==3.4.9.33
# opencv-contrib-python==3.4.9.33需要python版本大于3.6
pip install opencv-contrib-python==3.4.9.33
```


#### 验证opencv安装

打开python交互环境，输入以下代码：

```shell
import cv2
print(cv2.__version__)
```
