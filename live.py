# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:58:37 2020

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

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

def load(username):
    api = TikTokApi()
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
    cur.execute('refresh materialized view tiktok.videos_all_materialized')
    cur.execute('refresh materialized view tiktok.videos_materialized')
    conn.commit()

def histogram(username):
    plt.hist(np.log10(df['play_count']), density = True)
    plt.title('Play count distribution')
    plt.xlabel('log10(views)')
    plt.ylabel('Density')

def view_like(username):
    plt.scatter(np.log10(df['play_count']), np.log10(df['like_count']))
    plt.title('Plays vs. likes')
    plt.xlabel('log10(Plays)')
    plt.ylabel('log10(Likes)')

def make_overtime_plot(df, ax, x, y, y2):
    l1, = ax.plot(df[x], df[y], 'b-', label = y)
    ax2 = ax.twinx()
    l2, = ax2.plot(df[x], df[y2], 'r-', label = y2)
    ax.set_title(y2)
    ax.legend(handles = [l1, l2])

def most_recent(username):
    cur.execute('''
                select *,
                (fetch_time - create_time) as "Elapsed Time"
                 from tiktok.videos_all_materialized m
                where id = (
                    select id 
                    from tiktok.videos_all_materialized
                    where author = (%s)
                    order by create_time desc
                    limit 1
                )
                ''', [username])        
    overtime_df = pd.DataFrame(cur.fetchall(), 
                               columns = [desc[0] for desc in cur.description])
    overtime_df['VPL'] = overtime_df['play_count'] / overtime_df['like_count']
    fig, ax = plt.subplots(1,4, sharey = 'row')
    make_overtime_plot(overtime_df, ax[0], 'Elapsed Time', 'play_count', 'like_count')
    make_overtime_plot(overtime_df, ax[1], 'Elapsed Time', 'play_count', 'share_count')
    make_overtime_plot(overtime_df, ax[2], 'Elapsed Time', 'play_count', 'comment_count')
    make_overtime_plot(overtime_df, ax[3], 'Elapsed Time', 'play_count', 'VPL')
    plt.tight_layout()
    ax[0].set_ylabel('Views')
    ax[1].set_xlabel('Minutes since publication')
    print(f"Average VPL: {np.mean(overtime_df['VPL'])}")

username ='billnye' 
load_only = False
if load_only:
    load(username)
else:
    cur.execute('''SELECT *
    from tiktok.videos_materialized
    where author = (%s)''', [username])
    res=cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(res, columns = colnames)
    
    histogram(username)
    plt.figure()
    view_like(username)
    most_recent(username)

cur.close()
conn.close()
