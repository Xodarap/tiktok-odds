# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 15:02:47 2020

@author: bwsit
"""


import imgur
import qoves
import face
import pandas as pd
import os
import glob
import ntpath
import cv2
import numpy as np

def make_qoves_df(path):
    folder_name = os.path.basename((path[:-1]))
    album = imgur.create_album(folder_name)
    album_hash = album['data']['deletehash']
    album_id = album['data']['id']
    images = glob.glob(path + '*.jpg')
    
    df = pd.DataFrame()
    
    for image in images:
        title = os.path.splitext(ntpath.basename(image))[0]
        url, output = qoves.evaluate_path(image, title, album_hash)
        output['url'] = url
        output['album url'] = 'https://imgur.com/a/' + album_id
        df = df.append(output)
    
    df.to_csv(path + 'qoves.csv')

def ben_anal_single(folder, title):
    result = face.run_image(folder, 
                            title)
    output_folder = folder + 'output/'
    gray = cv2.cvtColor(result.edge_cropped, cv2.COLOR_GRAY2RGB) * 255
    cv2.imwrite(output_folder + title + ' edge cropped.jpg', gray)
    cv2.imwrite(output_folder + title + ' edge full.jpg', 
                cv2.cvtColor(result.edge_full, cv2.COLOR_GRAY2RGB) * 255)
    cv2.imwrite(output_folder + title + ' cropped.jpg', cv2.cvtColor(result.img_cropped, cv2.COLOR_BGR2RGB))
    cv2.imwrite(output_folder + title + ' full.jpg', cv2.cvtColor(result.img_full, cv2.COLOR_BGR2RGB))
    return [title, result.color_distance, result.wrinkle_percent]

def ben_anal(path):
    images = glob.glob(path + '*.jpg')
    results = []
    for image in images:
        title = os.path.splitext(ntpath.basename(image))[0]
        results.append(ben_anal_single(path, title))
    ben_df = pd.DataFrame(results, columns = ['Image', 'Color Distance', 'Wrinkle Percent'])
    ben_df.to_csv(path + 'ben.csv')

def run_all(path):
    ben_anal(path)
    make_qoves_df(path)

path = "D:/Documents/tiktok-live-graphs/makeup-bb/"
run_all(path)
