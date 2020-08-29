# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:43:42 2020

@author: bwsit
"""


import imgur
import requests
from bs4 import BeautifulSoup
import pandas as pd

def evaluate_image(url, title):
    payload = {'action': 'get_clarifai_tags_public',
               'image_url': url,
               'image_id' : ''}
    files = []
    headers = {}
    url = 'https://www.qoves.com/wp-admin/admin-ajax.php'
    response = requests.request("POST", url, headers=headers, data = payload, files = files)
    parsed = BeautifulSoup(response.text, features='lxml')
    links = parsed.find_all('a')
    
    def parse_link(link):
        ps = link.find_all('p')
        return [ps[0].text, ps[1].text]
    
    tab = [[title, *parse_link(l)] for l in links]
    return pd.DataFrame(tab, columns = ['Image', 'Flaw', 'Confidence'])

def evaluate_path(path, title, album_hash = '1t9Mpfemtvldk1a'):
    data = imgur.upload_image(album_hash, path)
    url = data['data']['link']
    return url, evaluate_image(url, title)

# url, output = evaluate_path("D:\\Documents\\tiktok-live-graphs\\makeup-overtime\\Maybelline control.jpg",
#                             'mbc')

# df = evaluate_image('https://i.imgur.com/dHKsyQw.jpg', 'Control')
    