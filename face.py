# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:41:31 2020

@author: Ben West
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import urllib.request as urlreq

def find_faces(src):
    # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    
    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"
    
    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        pass
        # print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")
    
    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)
    
    image_cropped = src #[y:(y+depth), x:(x+width)]
    image_template = image_cropped.copy()
    image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(src)
    
    # Print coordinates of detected faces
    # print("Faces:\n", faces)
    
    for face in faces:
        # save the coordinates in x, y, w, d variables
        (x,y,w,d) = face
        # Draw a white coloured rectangle around each face using the face's coordinates
        # on the "image_template" with the thickness of 2 
        y2 = y+d
        x2 = x+w
        cv2.rectangle(image_template,(x,y),(x2, y2),(255, 255, 255), 5)
    
    # plt.axis("off")
    # plt.imshow(image_template)
    # plt.title('Face Detection')
    # plt.figure()
    return faces, image_gray

def find_landmarks(image_gray, image_cropped, faces, title):
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    
    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "lbfmodel.yaml"
    
    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        pass
        # print("File exists")
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
    		# display landmarks on "image_cropped"https://www.qoves.com/wp-admin/admin-ajax.php
    		# with white colour in BGR and thickness 1
            cv2.circle(image_cropped, (x, y), 1, (255, 255, 255), 7)
    
    # plt.imshow(image_cropped)
    # plt.title(f'{title}')        
    # (x1,_) = landmarks[0][0][36]
    (x1,_) = landmarks[0][0][40]
    (x2,_) = landmarks[0][0][39]
    (_,y1) = landmarks[0][0][40]
    (x3,y12) = landmarks[0][0][41]
    (_,y2) = landmarks[0][0][38]
    y1 = max(y1, y12) #lower of two bottom eyelid
    x1 = int(x1)
    x2 = int(x2) + int(0.2*(x2 - x1))
    y1 = int(y1)
    y2 = int(y1) + int(y1 - y2)
    x3 = int(x3)
    y3 = int(y1 - 5 * (y1 - y2))
    y4 = y3 + int(y1- y2)
    cv2.rectangle(image_cropped, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(image_cropped, (x3, y3), (x1, y4), (255, 255, 255), 2)
    x5,y5 = landmarks[0][0][18]
    x6,y6 = landmarks[0][0][51]
    x5,y5,x6,y6 = map(int, [x5,y5,x6,y6])
    # plt.axis("off")
    # image_cropped = image_cropped[y5:y6,x5:x6]
    # plt.imshow(image_cropped)
    # plt.figure()
    return x1,y1,x2,y2,x3,y3,y4,image_cropped,image_cropped[y5:y6,x5:x6]

def find_edges(src2, title):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    src2 = cv2.GaussianBlur(src2, (3, 3), 3)
    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, 
                      borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, 
                      borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad = np.array(grad)
    threshold = np.mean(grad) + 1.3 * np.std(grad)
    grad[grad < threshold] = 0
    grad[grad >= threshold] = 1
    # plt.imshow(grad, cmap = 'gray')
    # plt.title(title)
    # plt.figure()
    return grad

class AnalResult():
    def __init__(self, img_path, img_full, img_cropped, edge_full, edge_cropped,
                 wrinkle_percent, color_distance):
        self.img_path = img_path
        self.img_full = img_full
        self.img_cropped = img_cropped
        self.edge_full = edge_full
        self.edge_cropped = edge_cropped
        self.wrinkle_percent = wrinkle_percent
        self.color_distance = color_distance

def run_image(folder, file_name):
    src = cv2.imread(folder + file_name + '.jpg')
    src2 = src.copy()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    
    faces, image_gray = find_faces(src)
    (x,y,w,d) = faces[0]
    face_pic = src2[y:y+d, x:x+w]
    x1,y1,x2,y2,x3,y3,y4,img_full, img_cropped = find_landmarks(image_gray, src, faces, file_name)
    relevant = src[y1:y2, x1:x2]
    grad = find_edges(face_pic, file_name)        
    relevant = grad[y1-y:y2-y, x1-x:x2-x]
    # plt.imshow(relevant, cmap = 'gray')
    # plt.title(file_name)
    # plt.figure()
    # plt.imshow(img_full)
    # plt.figure()
    # plt.imshow(img_cropped)
    # print(f"Fraction of {file_name} which is wrinkles: {np.sum(relevant) / relevant.size}")
    swatch1 = src[y1:y2, x1:x2]
    swatch2 = src[y4:y3, x3:x1]
    def get_avg(swatch):
        a = swatch.mean(axis=0).mean(axis=0)
        # plt.figure()
        # plt.imshow([[a.astype('int32')] * 5] * 5)
        return a
    a1 = get_avg(swatch1)
    a2 = get_avg(swatch2)
    # print(f"Color distance: {np.linalg.norm(a1-a2)}")
    return AnalResult(folder + file_name + '.jpg',
                      img_full, img_cropped, grad, relevant, 
                      np.sum(relevant) / relevant.size,
                      np.linalg.norm(a1-a2))

folder = "D:\\Documents\\tiktok-live-graphs\\makeup-overtime\\"
def run_folder(folder):
    # file_name = "covergirl end.jpg"
    product = 'Maybelline'
    results = []
    for stage in ['control', 'start', 'end']:
        results.append(run_image(folder, f'{product} {stage}'))
    return results


# def build_filters():
#     filters = []
#     ksize = 40
#     for theta in np.arange(0, np.pi, np.pi / 8):
#         for lamda in np.arange(0, enp.pi, np.pi/4):
#             kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#             kern /= 1.5*kern.sum()
#             filters.append(kern)
#     return filters
 
# def process(img, filters):
#     accum = np.zeros_like(img)
#     for kern in filters:
#         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
#         np.maximum(accum, fimg, accum)
#     return accum

# filters = build_filters()
# image_gray = cv2.GaussianBlur(image_gray, (5, 5), 5)
# res1 = process(image_gray, filters)
# plt.figure()
# plt.imshow(res1, cmap = 'gray')

# edges = cv2.Canny(image_gray,50,150)

# plt.figure()
# plt.imshow(edges,cmap = 'gray')
# res1 = cv2.resize(res1, (400, 400))
# cv2.imshow('result', res1)
# run_image(folder, file_name)
# src = cv2.imread("D:\\Pictures\\Best head shots\\lace2.jpg")
# src = cv2.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\control.jpg")
# src = cv2.imread("D:\\Documents\\tiktok-live-graphs\\makeup\\covergirl.jpg")
