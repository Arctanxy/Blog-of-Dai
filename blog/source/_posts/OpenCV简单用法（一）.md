---
title:      OpenCV简单用法（一）
subtitle:   图片的读取与转换
date:       2018-04-30
mathjax: true
cover: /img/OpenCV简单用法/Diablo.jpg
author: 
  nick: dalalaa
  link: https://github.com/Arctanxy
tags:
    - OpenCV
    - 图像处理
---



# 一、 OpenCV基本操作

OpenCV是一套使用C++编写的开源跨平台计算机视觉库，它提供了很简单的Python调用接口：


```python
import cv2
```

## 1. 图像的输入输出

### (1) imread()

使用imread()从文件中读入图像数据，其返回值是一个元素类型为unit8的三维数组（OpenCV中还提供了imwrite()函数用于写入图片数据）。

### (2) GUI工具

OpenCV提供了一些简单的GUI工具函数，如namedWindow(name)用于创建名为name的窗口，imshow()用于在窗口中显示图像。

### (3) waitKey()

watiKey(time)表示等待用户按键，如果time=0则表示永远等待。


```python
img = cv2.imread("Diablo.jpg")
print(type(img),img.shape,img.dtype)
cv2.namedWindow("Diablo")
cv2.imshow("Diablo",img)
cv2.waitKey(0)
```

    <class 'numpy.ndarray'> (1200, 1920, 3) uint8
    




    -1

![大菠萝](/img/OpenCV简单用法/Diablo.jpg)

## 2. 黑白转换


```python
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
cv2.namedWindow("Diablo_gray")
cv2.imshow("Diablo_gray",img_gray)
cv2.waitKey(0)
```

    (1200, 1920)
    




    -1

![黑白大菠萝](/img/OpenCV简单用法/Diablo.png)

## 3. 图像类型

### 通道

每个图片都可以看成像素点矩阵，图像中的每个像素点可能有多个通道，即包含多个颜色成分，例如用单通道可以表示灰度图像，而使用红绿蓝三个通道可以表示彩色图像，用4个通道可以表示带透明度的彩色图像。

如上面的图像，原本的彩色图像的shape为(1200,1920,3)，转换成黑白图像之后形状就变成了(1200,1920）。

### 比特

比特数又叫像素深度或图像深度，是一个像素点所占的总位数。

常说的256色图就是指像素点的比特数为8的图片，可以表示$2^8=256$种颜色



图像转换工作可以在读取文件的时候完成，imread()的第二个参数有如下选择：

### （1）IMREAD_ANYCOLOR

转换成8比特的图像，通道数由图像文件决定，但是4通道的图像会被自动转换成三通道。

### （2）IMREAD_ANYDEPTH

转换为单通道，比特数由图像文件决定。

### （3）IMREAD_COLOR

转换成三通道，8比特的图像

### （4）IMREAD_GRAYSCALE

转换成单通道，8比特的图像

### （5）IMREAD_UNCHANGED

使用图像文件的通道数和比特数

## 4. 图像输出

imwrite()将数组编码成指定的图像格式并写入文件，图像的格式由文件的扩展名决定。某些格式由额外的图像参数，例如JPEG格式的文件可以指定画质参数：


```python
img = cv2.imread("Diablo.jpg")
for quality in [30,60,90]:
    cv2.imwrite("Diablo%d.jpg" % quality,img,[cv2.IMWRITE_JPEG_QUALITY,quality])
```

## 5. 图像与字节序列的转换

imdecode()可以把图像文件数据解码成图像数组，imencode()则可以把图像数组编码成图像文件。


```python
import numpy as np 
with open("Diablo.jpg","rb") as f:
    jpg_str = f.read()

jpg_data = np.frombuffer(jpg_str,np.uint8)
img = cv2.imdecode(jpg_data,cv2.IMREAD_UNCHANGED)
img
```




    array([[[17, 19, 20],
            [ 5,  7,  8],
            [ 3,  5,  6],
            ..., 
            [ 2,  8,  7],
            [ 0,  0,  1],
            [ 4,  9, 10]],
    
           [[ 0,  2,  3],
            [ 4,  6,  7],
            [ 3,  5,  6],
            ..., 
            [ 3,  1,  1],
            [ 9,  6,  8],
            [ 3,  0,  2]],
    
           [[14, 16, 17],
            [ 8, 10, 11],
            [ 9, 11, 12],
            ..., 
            [21, 19, 19],
            [25, 22, 24],
            [12,  9, 11]],
    
           ..., 
           [[ 9, 28, 33],
            [54, 63, 67],
            [15, 18, 23],
            ..., 
            [ 8, 34, 40],
            [16, 19, 27],
            [18, 33, 36]],
    
           [[ 7, 23, 29],
            [27, 54, 58],
            [ 6, 34, 35],
            ..., 
            [25, 49, 55],
            [15, 25, 32],
            [11, 26, 29]],
    
           [[14,  4, 16],
            [35, 57, 62],
            [10, 46, 46],
            ..., 
            [28, 46, 53],
            [40, 56, 62],
            [13, 25, 29]]], dtype=uint8)




```python
from IPython.display import Image #将图片嵌入Jupyter Notebook
res,jpg_data = cv2.imencode(".jpg",img)
jpg_str = jpg_data.tobytes()
Image(data = jpg_str)
```




![jpeg](/img/OpenCV简单用法/output_15_0.jpeg)



# 二、OpenCV图像处理

## 1. 二维卷积

图像卷积的概念请参考[数字图像处理中的卷积](https://blog.csdn.net/chaipp0607/article/details/72236892?locationNum=9&fps=1)，常见的几种功能的卷积核有如下几种：

### (1)模糊化


```python
# 平滑均值滤波卷积核

A = np.array([[1,1,1],
             [1,1,1],
             [1,1,1]])

# 高斯平滑卷积核

B = np.array([[1,2,1],
            [2,2,2],
            [1,2,1]])
```

使用高斯卷积核处理之后的图像：


```python
import matplotlib.pyplot as plt 
%matplotlib inline
kernel = B/14#卷积核的所有元素之和最好为1，否则会出现过亮或者过暗的情况
src = cv2.imread("Diablo.jpg")
dst = cv2.filter2D(src,-1,kernel)
plt.figure(figsize=(20,35))
plt.imshow(dst[:,:,::-1])#matplotlib 和opencv的颜色是相反的
```




    <matplotlib.image.AxesImage at 0x10ae5e48>




![png](/img/OpenCV简单用法/output_22_1.png)


……效果不太明显，多模糊几次试试：


```python
kernel = B/14#卷积核的所有元素之和最好为1，否则会出现过亮或者过暗的情况
src = cv2.imread("Diablo.jpg")
dst = cv2.filter2D(src,-1,kernel)
for i in range(20):
    dst = cv2.filter2D(dst,-1,kernel)
plt.figure(figsize=(20,35))
plt.imshow(dst[:,:,::-1])
```




    <matplotlib.image.AxesImage at 0x119bf780>




![png](output_24_1.png)


整个图片都朦胧起来了

### (2) 锐化


```python
C = np.array([[-1,-1,-1],
             [-1,9,-1],
             [-1,-1,-1]])

kernel = C
src = cv2.imread("Diablo.jpg")
dst = cv2.filter2D(src,-1,kernel)
plt.figure(figsize=(20,35))
plt.imshow(dst[:,:,::-1])
```




    <matplotlib.image.AxesImage at 0xbdef7b8>




![png](/img/OpenCV简单用法/output_27_1.png)


### (3) 边缘检测


```python
D = np.array([[1,1,1],
             [1,-8,1],
             [1,1,1]])

kernel = D
src = cv2.imread("Diablo.jpg")
dst = cv2.filter2D(src,-1,kernel)
plt.figure(figsize=(20,35))
plt.imshow(dst[:,:,::-1])
```




    <matplotlib.image.AxesImage at 0xc0e63c8>




![png](/img/OpenCV简单用法/output_29_1.png)

