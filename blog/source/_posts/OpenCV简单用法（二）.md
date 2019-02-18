---
title:      OpenCV简单用法（二）
subtitle:   识别人脸与眼睛
date:       2018-04-30
mathjax: true
author: 
  nick: dalalaa
  link: https://github.com/Arctanxy
cover: /img/OpenCV简单用法/Atreus.jpg
tags:
    - OpenCV
    - 图像处理
---

# 最简单的人脸识别

还是用那张大菠萝的图片，这里需要用到两个文件：haarcascade_frontalface_default.xml和haarcascade_eye.xml，一个是找人脸的，一个是找眼睛的。

```python
import cv2 
import numpy as np 

face_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_eye.xml')

#cap = cv2.VideoCapture(1)

#while True:
img = cv2.imread("H:/learning_notes/study/opencv/Diablo.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#找到脸之后画个长方形框出来
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)#在脸的范围找眼镜
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)

#cap.release()
#cv2.destroyAllWindows()

```
运行以下看看结果：

![识别结果](/img/OpenCV简单用法/face.png)

只有圣骑士的脸被识别出来了，看来只有正面人脸才能识别出来，大菠萝虽然也是正脸，可惜长得太丑了。至于眼睛，受图片清晰度所限，是找不到了。


那么我们换一张年轻的战神的照片试试:

![年轻的战神](/img/OpenCV简单用法/Atreus.jpg)

识别结果：

![戴眼镜的战神](/img/OpenCV简单用法/face_atreus.png)

