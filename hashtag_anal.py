# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:41:29 2020

@author: bwsit
"""



import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TikTokApi import TikTokApi
import datetime
import time
import json
from dotenv import load_dotenv
load_dotenv()
import os

conn=psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()

def vid_count(tag, api):
    res = api.getHashtagObject(tag)
    cid = int(res['challengeInfo']['challenge']['id'])
    vid_count = res['challengeInfo']['stats']['videoCount'] 
    view_count = res['challengeInfo']['stats']['viewCount']
    cur.execute('''
                INSERT INTO tiktok.tag_data(
            	name, video_count, view_count, challenge_id, fetch_time)
            	VALUES (%s, %s, %s, %s, %s);
                ''', (tag, vid_count, view_count, cid, str(datetime.datetime.now())))
    conn.commit()
    vid_view_ratio="{:,.2f}".format(float(view_count)/float(vid_count))
    print("%s\t%s\t%s\t%s\t" % (tag, vid_view_ratio,vid_count, view_count))
api = TikTokApi()
'''
vid_count('art', api)
vid_count('collage', api)
vid_count('collageart', api)
vid_count('handmade', api)
vid_count('crafty', api)
vid_count('diy', api)
'''
vid_count('tiktokalgorithm', api)
vid_count('algorithm', api)
vid_count('statistics', api)
vid_count('algorithim', api)
vid_count('tiktokhack', api)
vid_count('tiktokhacks', api)
vid_count('algorithmhack', api)