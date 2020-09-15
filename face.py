# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:41:31 2020

@author: Ben West
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    # if multiple faces, choose the biggest
    faces = sorted(faces, key = lambda x: -x[3])
    # plt.imshow(image_template)
    # plt.title('Face Detection')
    # plt.figure()
    return faces, image_gray

class Landmark():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    
    def to_tuple(self):
        return (self.x, self.y)

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
    # cv2.rectangle(image_cropped, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.rectangle(image_cropped, (x3, y3), (x1, y4), (255, 255, 255), 2)
    x5,y5 = landmarks[0][0][18]
    x6,y6 = landmarks[0][0][51]
    x5,y5,x6,y6 = map(int, [x5,y5,x6,y6])
    def build_eye(l1, l2, m, u, left_side):
        comp_fn = max if left_side else min
        comp_amount = 1 if left_side else -1
        (l1x, l1y) = l1
        (l2x, l2y) = l2
        (mx, my) = m
        (ux, uy) = u
        left_x = comp_fn(l1x, l2x)
        right_x = mx + comp_amount * (0.2 * abs(left_x - mx))
        top_y = max(l1y, l2y) + 0.3 * abs(max(l1y, l2y) - uy)
        bottom_y = top_y + (top_y - uy)
        return (Landmark(left_x, top_y), Landmark(right_x, bottom_y))
    def build_cheek(eye1, eye2, nose_point, left):
        (x1, y1) = eye1.to_tuple()
        (x2, y2) = eye2.to_tuple()
        (_, nose_y) = nose_point
        top_y = nose_y
        bottom_y = nose_y + (y2-y1)
        factor = 1 if left else -1
        inner_x = x1 - factor * abs(x2-x1)
        outer_x = inner_x + factor * abs(x2-x1)
        return (Landmark(inner_x, top_y), Landmark(outer_x, bottom_y))
    land_array = landmarks[0][0]
    landmark_dictionary = {
        'left eye square': build_eye(land_array[41], land_array[40],
                                     land_array[39], land_array[38],
                                     True),
        'right eye square': build_eye(land_array[46], land_array[47],
                                     land_array[42], land_array[43],
                                     False),
        'left full square': (Landmark(*land_array[17]), Landmark(*land_array[51])),
        'right full square': (Landmark(*land_array[51]), Landmark(*land_array[26])),
        'all': land_array
        }
    landmark_dictionary['left cheek square'] = build_cheek(*landmark_dictionary['left eye square'],
                                                            land_array[30], True)
    landmark_dictionary['right cheek square'] = build_cheek(*landmark_dictionary['right eye square'],
                                                            land_array[30], False)
    (p1, p2) = landmark_dictionary['left eye square']
    cv2.rectangle(image_cropped, p1.to_tuple(), p2.to_tuple(), (255, 255, 255), 2)
    (p1, p2) = landmark_dictionary['right eye square']
    cv2.rectangle(image_cropped, p1.to_tuple(), p2.to_tuple(), (255, 255, 255), 2)
    (p1, p2) = landmark_dictionary['left cheek square']
    cv2.rectangle(image_cropped, p1.to_tuple(), p2.to_tuple(), (255, 255, 255), 2)
    (p1, p2) = landmark_dictionary['right cheek square']
    cv2.rectangle(image_cropped, p1.to_tuple(), p2.to_tuple(), (255, 255, 255), 2)
    # plt.axis("off")
    # image_cropped = image_cropped[y5:y6,x5:x6]
    # plt.imshow(image_cropped)
    # plt.figure()
    return x1,y1,x2,y2,x3,y3,y4,image_cropped,image_cropped[y5:y6,x5:x6],landmark_dictionary

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
    # grad[grad < threshold] = 0
    # grad[grad >= threshold] = 1
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

def eval_squares(src, grad, offset_x, offset_y, eye_square, cheek_square):
    (p1, p2) = eye_square
    (xa, y1) = p1.to_tuple()
    (xb, y2) = p2.to_tuple()
    (p1, p2) = cheek_square
    (xa2, y3) = p1.to_tuple()
    (xb2, y4) = p2.to_tuple()
    x1 = min(xa, xb)
    x2 = max(xa, xb)
    x3 = min(xa2, xb2)
    x4 = max(xa2, xb2)
    eg = get_square(grad, eye_square, offset_x, offset_y)
    cg = get_square(grad, cheek_square, offset_x, offset_y)
    # eg = grad[y1-offset_y:y2-offset_y, x1-offset_x:x2-offset_x]
    # cg = grad[y3-offset_y:y4-offset_y, x3-offset_x:x4-offset_x]
    swatch1 = src[y1:y2, x1:x2]
    swatch2 = src[y3:y4, x3:x4]
    return eg, cg, swatch1, swatch2

def get_square(image, pair, offset_x = 0, offset_y = 0):
    (p1, p2) = pair
    (xa, ya) = p1.to_tuple()
    (xb, yb) = p2.to_tuple()
    (y1, y2) = (min(ya, yb) - offset_y, max(ya, yb) - offset_y)    
    (x1, x2) = (min(xa, xb) - offset_x, max(xa, xb) - offset_x)
    return image[y1:y2, x1:x2]
def get_avg(swatch):
    return swatch.mean(axis=0).mean(axis=0)
def square_to_points(p1, p2):
    (x1, y1) = p1.to_tuple()
    (x2, y2) = p2.to_tuple()
    return x1, y1, x2, y2

def evaluate_half(src, grad, x, y, eye_square, cheek_square):
    eg, cg, es, cs = eval_squares(src, grad, x, y, eye_square, cheek_square)
    return {
        'eye wrinkle percent': np.sum(eg) / eg.size,
        'cheek wrinkle percent': np.sum(cg) / cg.size,
        'color distance': np.linalg.norm(get_avg(es) - get_avg(cs))
        }
def run_image(folder, file_name, show = False):
    src = cv2.imread(folder + file_name + '.jpg')
    src2 = src.copy()
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    
    faces, image_gray = find_faces(src)
    x1,y1,x2,y2,x3,y3,y4,img_full, img_cropped,landmark_dictionary = find_landmarks(image_gray, src, faces, file_name)

    (x,y,w,d) = faces[0]
    chin_y = int(landmark_dictionary['all'][8][1])
    d = max(d, chin_y-y) # make sure pic goes to bottom of chin
    face_pic = src2[y:y+d, x:x+w]
    grad = find_edges(face_pic, file_name)        


    sub_pic = lambda key: get_square(src, landmark_dictionary[key])
    grad_pic = lambda key: get_square(grad, landmark_dictionary[key], x, y)
    
    if show:
        fig, axs = plt.subplots(3,3)
        axs[0,1].imshow(img_full)
        axs[1,0].imshow(sub_pic('left full square'))
        axs[1,1].imshow(sub_pic('left eye square'))
        axs[1,2].imshow(sub_pic('left cheek square'))
        axs[2,0].imshow(sub_pic('right full square'))
        axs[2,1].imshow(sub_pic('right eye square'))
        axs[2,2].imshow(sub_pic('right cheek square'))
        
       
        fig, axs = plt.subplots(3,3)
        axs[0,1].imshow(grad,cmap = 'gray')
        axs[1,0].imshow(grad_pic('left full square'))
        axs[1,1].imshow(grad_pic('left eye square'))
        axs[1,2].imshow(grad_pic('left cheek square'))
        axs[2,0].imshow(grad_pic('right full square'))
        axs[2,1].imshow(grad_pic('right eye square'))
        axs[2,2].imshow(grad_pic('right cheek square'))
    # print(f"Color distance: {np.linalg.norm(a1-a2)}")
    return {'image path': folder + file_name + '.jpg',
            'image full': img_full,
            'gradient full': grad,
            'landmarks': landmark_dictionary,
            'left results': evaluate_half(src, grad, x, y, landmark_dictionary['left eye square'],
                                              landmark_dictionary['left cheek square']),
            'right results': evaluate_half(src, grad, x, y, landmark_dictionary['right eye square'],
                                              landmark_dictionary['right cheek square']),
            'sub pic': sub_pic,
            'grad pic': grad_pic}

folder = "D:\\Documents\\tiktok-live-graphs\\makeup-followers\\naima\\"
def run_folder(folder):
    # file_name = "covergirl end.jpg"
    product = 'Maybelline'
    results = []
    for stage in ['control', 'start', 'end']:
        results.append(run_image(folder, f'{product} {stage}'))
    return results

run_image(folder, 'Covergirl No Makeup', True)
# results = run_image('D:\\Documents\\tiktok-live-graphs\\mmmmarkie\\', 
#                     'foundation half primer', True)
# print(results)
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
