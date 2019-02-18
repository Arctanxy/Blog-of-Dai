---
title:      线性回归的矩阵运算
date:       2018-04-05
mathjax: true
author: 
  nick: dalalaa
  link: https://github.com/Arctanxy
tags: 
    - 机器学习
    - 线性回归
---



# 岭回归的矩阵运算


## 普通线性回归的矩阵运算


在训练数据
（1）各列数据线性独立
（2）样本数量多于特征数量
的前提下，可以使用矩阵形式计算线性回归的系数
[参考](http://bourneli.github.io/linear-algebra/2016/03/03/linear-algebra-04-ATA-inverse.html)

### 想象中的推导过程

$$ Xw = Y $$

$$ X^TXw = X^TY$$

$$ w = (X^TX)^{-1}X^TY$$

步骤很简单，但是其实这是一种野路子的算法，因为这一步的原始方程$Xw = Y$不是一个能直接使用的方程，这个方程可以通过坐标投影得到，只能作为一个经验公式使用，具体可以参见https://zhuanlan.zhihu.com/p/22757336。

### 真实的推导过程

推导之前需要了解一些关于矩阵范数的知识。

### 矩阵范数

矩阵的L2范数

$$||A||^2_2 = \lambda$$

其中$\lambda$是矩阵$A^TA$的最大特征值。

因此，对于一维矩阵（向量）$a$而言，其L2范数就是$a^Ta$，因为$a^Ta$是一个标量，所以$a^Ta$的值就是其最大特征值。

### 矩阵微分

当$X$，$\beta$是一维矩阵（向量）时，

$$ \frac{\partial \beta^TX}{X} = \beta$$

$$ \frac{\partial X^TX}{\partial X} = X$$

$$ \frac{X^TAX}{\partial X} = (A + A^T)X$$


[矩阵微分公式](https://blog.csdn.net/u010976453/article/details/54381248)

回归的目标是最小化

$$min(||Y-XW||^2_2)$$

### 计算过程


$$ F = ||Y-XW||^2_2 = (Y-XW)^T(Y-XW)$$

$$ F = (Y^T - W^TX^T)(Y-XW)$$

$$ F = Y^TY - Y^TXW - W^TX^TY + W^TX^TXW$$

对F进行求导得到：

$$ \frac{\partial F}{\partial W} = - Y^TX - Y^TX + (X^TX + X^TX)W $$

$$ \frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW $$

令$\frac{\partial F}{\partial W} = 0$，有：

$$\frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW = 0$$

$$X^TXW = Y^TX$$

$$W = (X^TX)^{-1} Y^TX$$

### 利用矩阵的迹的性质推导
参见斯坦福大学cs229课程笔记
依赖公式：

$$ tr(a) = a$$

$$ trAB = trBA$$

$$ trA = trA^T$$

$$ \nabla_AtrAB = B^T$$

$$ \nabla_{A^T} trABA^TC = B^TA^TC^T + BA^TC$$


其中a是一个实数

也可以写成

$$\frac{\partial(trAB)}{\partial A} = B^T$$

$$\frac{\partial(trABA^TC)}{A^T} = B^TA^TC^T + BA^TC$$

但是这个推导过程用梯度形式更好书写，所以下面还是遵照原笔记中的梯度公式进行推导

推导过程：

$$ F = ||Y-XW||^2_2 = (Y-XW)^T(Y-XW)$$

直接求导

$$ \nabla_W F(W) = \nabla_W(Y-XW)^T(Y-XW) $$

$$  = \nabla_W(Y^TY - Y^TXW - W^TX^TY + W^TX^TXW)$$

因为括号中的所有项的计算结果是一个实数，所以：

$$ \nabla_W F(W) = \nabla_W tr(Y^TY - Y^TXW - W^TX^TY + W^TX^TXW)$$

因为$trA = trA^T$，可以化简为：

$$ \nabla_W F(W) = \nabla_W (Y^TY - 2trY^TXW + trW^TX^TXW)$$

再利用公式$\nabla_{A^T} trABA^TC = B^TA^TC^T + BA^TC$进行化简，令$A^T = W$，$B = B^T = X^TX$，$C = I$:

$$ \nabla_W F(W) = \nabla_W (Y^TY - 2trY^TXW + trABA^TC)$$

展开为：

$$ \nabla_W F(W) = \nabla_W (Y^TY - 2trY^TXW) + X^TXW + X^TXW$$

$$ \nabla_W F(W) = \nabla_W (-2trY^TXW) + 2X^TXW$$

另外根据公式$\nabla_AtrAB = B^T$以及公式$trAB = trBA$可以得到$\nabla_AtrBA = B^T$，所以上式可以继续化简为：

$$ \nabla_W F(W) = -2X^TY + 2X^TXW$$

令梯度为0，即可得到$minF(W)$所对应的$W$：

$$2X^TY = 2X^TXW$$

$$W = (X^TX)^{-1}X^TY$$

推导完毕


## 岭回归的矩阵运算


岭回归的矩阵运算推导跟线性回归类似，只是多了一个范数求导的过程


目标：

$$min(||Y-XW||^2_2 + \lambda||W||^2_2)$$

#### 计算过程


$$ F = ||Y-XW||^2_2 + \lambda||W||^2_2 = (Y-XW)^T(Y-XW) + \lambda W^TW$$

$$ F = (Y^T - W^TX^T)(Y-XW) + \lambda W^TW$$

$$ F = Y^TY - Y^TXW - W^TX^TY + W^TX^TXW + \lambda W^TW$$

对F进行求导得到：



$$ \frac{\partial F}{\partial W} = - Y^TX - Y^TX + (X^TX + X^TX)W + \lambda W$$

$$ \frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW + \lambda W$$

令$\frac{\partial F}{\partial W} = 0$，有：

$$\frac{\partial F}{\partial W} = -2 Y^TX + 2X^TXW + \lambda W = 0$$

$$(X^TX + \frac{\lambda I}{2})W = Y^TX$$

$$W = (X^TX + \frac{\lambda I}{2})^{-1}Y^TX$$

[推导地址](https://blog.csdn.net/computerme/article/details/50486937)
