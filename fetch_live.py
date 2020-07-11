# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:47:28 2020

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

def load(username):
    api = TikTokApi()
    print(f'Fetching {username}')
    try:
        tiktoks = api.byUsername(username, count=50)
    except Exception as e:
        print(e)
    t_time=str(datetime.datetime.now())
    print(len(tiktoks))
    print(t_time)
    for t in tiktoks:
        new_t=json.dumps(t)
        cur.execute('INSERT INTO tiktok (time,json) VALUES (%s,%s)', (t_time,new_t))
    conn.commit()
    print('Saved')

def refresh_views():
    cur.execute('refresh materialized view tiktok.videos_all_materialized')
    cur.execute('refresh materialized view tiktok.videos_materialized')
    conn.commit()
    print('Views Refreshed')

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

usernames = ['billnye', 'benthamite', 'amarchenkova']
for username in usernames:
    load(username)

#refresh_views()

cur.close()
conn.close()