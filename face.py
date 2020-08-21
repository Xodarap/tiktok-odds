# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:41:31 2020

@author: bwsit
"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request as urlreq
from pylab import rcParams

# src = cv.imread("D:\\Pictures\\Best head shots\\lace2.jpg")
# src = cv.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\control.jpg")
# src = cv.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\covergirl.jpg")
src = cv.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\age rewind.jpg")
# src = cv.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\loreal.jpg")
src2 = src.copy()
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# plt.imshow(grad,  cmap='gray')
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"

# chech if file is in working directory
if (haarcascade in os.listdir(os.curdir)):
    print("File exists")
else:
    # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade)

# rs = cv2.resize(src, (950, 500))
# x, y, width, depth = 0, 00, 950, 500
image_cropped = src #[y:(y+depth), x:(x+width)]
image_template = image_cropped.copy()
image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(src)

# Print coordinates of detected faces
print("Faces:\n", faces)

for face in faces:
#     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2 
    y2 = y+d
    x2 = x+w
    cv2.rectangle(image_template,(x,y),(x2, y2),(255, 255, 255), 5)

plt.axis("off")
plt.imshow(image_template)
plt.title('Face Detection')
plt.figure()
# (x,y,w,d) = faces[0]
# image_gray = image_gray[x:x+w, y:y+w]
# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "lbfmodel.yaml"

# check if file is in working directory
if (LBFmodel in os.listdir(os.curdir)):
    print("File exists")
else:
    # download picture from url and save locally as lbfmodel.yaml, < 54MB
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image_gray, np.array([faces[0]]))

eyes = [np.array([landmarks[0][0][36:42]])]

for landmark in landmarks:
    for x,y in landmark[0]:
		# display landmarks on "image_cropped"
		# with white colour in BGR and thickness 1
        # print((x,y))
        cv2.circle(image_cropped, (x, y), 1, (255, 255, 255), 7)
plt.imshow(image_cropped)
plt.title('Landmarks')        
# (x1,_) = landmarks[0][0][36]
(x1,_) = landmarks[0][0][40]
(x2,_) = landmarks[0][0][39]
(_,y1) = landmarks[0][0][40]
(_,y12) = landmarks[0][0][41]
(_,y2) = landmarks[0][0][38]
y1 = max(y1, y12) #lower of two bottom eyelid
x1 = int(x1)
x2 = int(x2) + int(0.2*(x2 - x1))
y1 = int(y1)
y2 = int(y1) + int(y1 - y2)
cv2.rectangle(image_cropped,(x1, y1), (x2, y2),(255, 255, 255), 2)
plt.axis("off")
plt.imshow(image_cropped)
# erfsaw
plt.figure()
scale = 1
delta = 0
ddepth = cv.CV_16S
# src2 = src #cv.imread("D:\\Pictures\\Best head shots\\lace2.jpg")
src2 = cv.GaussianBlur(src2, (3, 3), 0)
gray = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
def above(n, array):
    def row(r):
        return [10 if v > n else 0 for v in r]
    return [row(r) for r in array]
grad = np.array(grad)
# grad = (grad > np.mean(grad)) * grad
threshold = np.mean(grad) + np.std(grad)
grad[grad < threshold] = 0
grad[grad >= threshold] = 1
# cv2.rectangle(grad, (x1, y1), (x2, y2),(1, 0, 0), 5)
# grad = above(np.mean(grad), grad)
relevant = grad[y1:y2, x1:x2]
plt.imshow(grad)
plt.title('Sobel Edge Detection')
plt.figure()
plt.imshow(relevant)
plt.title('Edge Detection - Zoomed In')
(l, w) = relevant.shape
print(np.sum(relevant) / (l * w))