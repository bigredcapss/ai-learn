#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像基本操作 - OpenCV实战教程
环境配置地址：
- Anaconda: https://www.anaconda.com/download/
- Python_whl: https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
- eclipse: 按照自己的喜好，选择一个能debug就好
"""

import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def cv_show(name, img):
    """显示图像的辅助函数"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("=== OpenCV图像基本操作演示 ===\n")
    
    # 1. 数据读取-图像
    print("1. 读取图像...")
    # cv2.IMREAD_COLOR：彩色图像
    # cv2.IMREAD_GRAYSCALE：灰度图像
    img = cv2.imread('cat.jpg')
    
    if img is None:
        print("错误：无法读取图像文件 'cat.jpg'")
        return
    
    print(f"图像形状: {img.shape}")
    print(f"图像类型: {type(img)}")
    print(f"图像大小: {img.size}")
    print(f"图像数据类型: {img.dtype}")
    
    # 显示图像
    print("\n显示彩色图像...")
    cv_show('image', img)
    
    # 读取灰度图像
    print("\n读取灰度图像...")
    img_gray = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
    print(f"灰度图像形状: {img_gray.shape}")
    
    # 显示灰度图像
    print("显示灰度图像...")
    cv2.imshow('gray_image', img_gray)
    cv2.waitKey(10000)  # 等待10秒
    cv2.destroyAllWindows()
    
    # 保存图像
    print("\n保存图像...")
    cv2.imwrite('mycat.png', img_gray)
    print("图像已保存为 'mycat.png'")
    
    # 2. 数据读取-视频
    print("\n2. 读取视频...")
    vc = cv2.VideoCapture('test.mp4')
    
    # 检查是否打开正确
    if vc.isOpened():
        open_flag, frame = vc.read()
        print("视频文件打开成功")
        
        # 读取视频帧
        while open_flag:
            ret, frame = vc.read()
            if frame is None:
                break
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('result', gray)
                if cv2.waitKey(100) & 0xFF == 27:  # ESC键退出
                    break
        vc.release()
        cv2.destroyAllWindows()
    else:
        print("错误：无法打开视频文件 'test.mp4'")
    
    # 3. 截取部分图像数据
    print("\n3. 截取部分图像数据...")
    img = cv2.imread('cat.jpg')
    cat = img[0:50, 0:200]
    cv_show('cat', cat)
    
    # 4. 颜色通道提取
    print("\n4. 颜色通道提取...")
    b, g, r = cv2.split(img)
    print(f"红色通道形状: {r.shape}")
    
    # 合并通道
    img_merged = cv2.merge((b, g, r))
    print(f"合并后图像形状: {img_merged.shape}")
    
    # 只保留R通道
    print("显示红色通道...")
    cur_img = img.copy()
    cur_img[:, :, 0] = 0  # 蓝色通道置0
    cur_img[:, :, 1] = 0  # 绿色通道置0
    cv_show('R', cur_img)
    
    # 只保留G通道
    print("显示绿色通道...")
    cur_img = img.copy()
    cur_img[:, :, 0] = 0  # 蓝色通道置0
    cur_img[:, :, 2] = 0  # 红色通道置0
    cv_show('G', cur_img)
    
    # 只保留B通道
    print("显示蓝色通道...")
    cur_img = img.copy()
    cur_img[:, :, 1] = 0  # 绿色通道置0
    cur_img[:, :, 2] = 0  # 红色通道置0
    cv_show('B', cur_img)
    
    # 5. 边界填充
    print("\n5. 边界填充...")
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, 
                                  borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, 
                                cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, 
                                   cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, 
                             cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, 
                                 cv2.BORDER_CONSTANT, value=0)
    
    # 显示边界填充效果
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
    plt.subplot(232), plt.imshow(cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB)), plt.title('复制法')
    plt.subplot(233), plt.imshow(cv2.cvtColor(reflect, cv2.COLOR_BGR2RGB)), plt.title('反射法')
    plt.subplot(234), plt.imshow(cv2.cvtColor(reflect101, cv2.COLOR_BGR2RGB)), plt.title('反射法101')
    plt.subplot(235), plt.imshow(cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB)), plt.title('外包装法')
    plt.subplot(236), plt.imshow(cv2.cvtColor(constant, cv2.COLOR_BGR2RGB)), plt.title('常量法')
    plt.tight_layout()
    plt.show()
    
    print("边界填充方法说明：")
    print("- BORDER_REPLICATE：复制法，复制最边缘像素")
    print("- BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制")
    print("- BORDER_REFLECT_101：反射法，以最边缘像素为轴，对称")
    print("- BORDER_WRAP：外包装法")
    print("- BORDER_CONSTANT：常量法，常数值填充")
    
    # 6. 数值计算
    print("\n6. 数值计算...")
    img_cat = cv2.imread('cat.jpg')
    img_dog = cv2.imread('dog.jpg')
    
    if img_cat is not None and img_dog is not None:
        img_cat2 = img_cat + 10
        print("原始图像前5个像素值:", img_cat[:5, :, 0])
        print("加10后前5个像素值:", img_cat2[:5, :, 0])
        print("直接相加前5个像素值:", (img_cat + img_cat2)[:5, :, 0])
        print("cv2.add相加前5个像素值:", cv2.add(img_cat, img_cat2)[:5, :, 0])
    
    # 7. 图像融合
    print("\n7. 图像融合...")
    if img_cat is not None and img_dog is not None:
        print(f"猫图像形状: {img_cat.shape}")
        print(f"狗图像形状: {img_dog.shape}")
        
        # 调整狗图像大小以匹配猫图像
        img_dog = cv2.resize(img_dog, (500, 414))
        print(f"调整后狗图像形状: {img_dog.shape}")
        
        # 图像融合
        res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img_cat, cv2.COLOR_BGR2RGB)), plt.title('猫图像')
        plt.subplot(132), plt.imshow(cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)), plt.title('狗图像')
        plt.subplot(133), plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)), plt.title('融合结果')
        plt.tight_layout()
        plt.show()
    
    # 8. 图像缩放
    print("\n8. 图像缩放...")
    if img is not None:
        # 放大4倍
        res = cv2.resize(img, (0, 0), fx=4, fy=4)
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
        plt.subplot(122), plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)), plt.title('放大4倍')
        plt.tight_layout()
        plt.show()
        
        # 高度放大3倍
        res = cv2.resize(img, (0, 0), fx=1, fy=3)
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('原始图像')
        plt.subplot(122), plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)), plt.title('高度放大3倍')
        plt.tight_layout()
        plt.show()
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()
