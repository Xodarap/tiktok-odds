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
from itertools import chain
import functools as f

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
    
    df['Confidence'] = df['Confidence'].astype('float')
    table = pd.pivot_table(df, values = 'Confidence', columns ='Flaw', index = 'Image')
    table.to_csv(path + 'qoves_pivot.csv')
    df.to_csv(path + 'qoves.csv')

def ben_anal_single(folder, title):
    result = face.run_image(folder, 
                            title)
    output_folder = folder + 'output/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    styles = [{'title': 'plain', 'fn': result['sub pic'], 
               'conversion': lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2RGB)},
              {'title': 'edge', 'fn': result['grad pic'], 
               'conversion': lambda i: cv2.cvtColor(i, cv2.COLOR_GRAY2RGB) * 255}]
    result_table = []
    for style in styles:
        for side in ['left', 'right']:
            side_results = [title, side]
            for part in ['eye', 'cheek', 'full']:
                subtitle = f'{side} {part} square'
                image = style['fn'](subtitle)
                cv2.imwrite(output_folder + f'{title} {subtitle} {style["title"]}.jpg', 
                            style['conversion'](image))
                if part != 'full':
                    side_results.append(result[side + ' results'][part + ' wrinkle percent'])
            
            side_results.append(result[side + ' results']['color distance'])
            if style['title'] == 'plain':
                result_table.append(side_results)
        full_key = 'image' if style['title'] =='plain' else 'gradient'
        cv2.imwrite(output_folder + f'{title} {style["title"]}.jpg', 
                            style['conversion'](result[full_key + ' full']))
    # gray = cv2.cvtColor(result.edge_cropped, cv2.COLOR_GRAY2RGB) * 255
    # cv2.imwrite(output_folder + title + ' edge cropped.jpg', gray)
    # cv2.imwrite(output_folder + title + ' edge full.jpg', 
    #             cv2.cvtColor(result.edge_full, cv2.COLOR_GRAY2RGB) * 255)
    # cv2.imwrite(output_folder + title + ' cropped.jpg', cv2.cvtColor(result.img_cropped, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(output_folder + title + ' full.jpg', cv2.cvtColor(result.img_full, cv2.COLOR_BGR2RGB))
    return result_table

def make_summary(ben_df):
    sort_order = {'No Makeup': 0, 'Start': 1, 'Midday': 2, 'End': 3}
    ben_df['time_sort_order'] = ben_df['Time'].map(lambda k: sort_order[k])
    summary_table = ben_df.groupby(['Product', 'Time']).mean()
    summary_table.sort_values(by = ['Product', 'time_sort_order'], inplace = True)
    summary_table.to_csv(path + 'ben_summary.csv')
    

def ben_anal(path):
    images = glob.glob(path + '*.jpg')
    results = []
    for image in images:
        title = os.path.splitext(ntpath.basename(image))[0]
        #todo: just +?
        results = f.reduce(lambda x, y: x+y, [ben_anal_single(path, title)], results)
    ben_df = pd.DataFrame(results, 
                          columns = ['Image', 'Face Side', 
                                     'Eye Wrinkle Percent', 
                                     'Cheek Wrinkle Percent', 'Color Distance'])
    names = ben_df['Image'].str.split(' ').str
    ben_df['Time'] = names[1:].str.join(' ')
    ben_df['Product'] = names[0]
    ben_df.to_csv(path + 'ben.csv')
    make_summary(ben_df)

# [[x]] -> [[x]] -> [[x]]
def run_all(path):
    ben_anal(path)
    # make_qoves_df(path)

path = "D:/Documents/tiktok-live-graphs/makeup-followers/naima/"
run_all(path)
# df = pd.read_csv(path + 'qoves.csv')
# table = pd.pivot_table(df, values = 'Confidence', columns ='Flaw', index = 'Image')
# table.to_csv(path + 'qoves_pivot.csv')