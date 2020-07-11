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
    print('Saved')
    cur.execute('refresh materialized view tiktok.videos_all_materialized')
    cur.execute('refresh materialized view tiktok.videos_materialized')
    conn.commit()
    print('Views Refreshed')

def histogram(df):
    plt.hist(np.log10(df['play_count']), density = True)
    plt.title('Play count distribution')
    plt.xlabel('log10(views)')
    plt.ylabel('Density')

def view_like(df):
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
    overtime_df['Engagements'] = np.sum(overtime_df[['like_count', 'share_count', 'comment_count']])
    overtime_df['EPL'] = overtime_df['play_count'] / overtime_df['Engagements']
    fig, ax = plt.subplots(1,4, sharey = 'row')
    make_overtime_plot(overtime_df, ax[0], 'Elapsed Time', 'play_count', 'like_count')
    make_overtime_plot(overtime_df, ax[1], 'Elapsed Time', 'play_count', 'share_count')
    make_overtime_plot(overtime_df, ax[2], 'Elapsed Time', 'play_count', 'comment_count')
    make_overtime_plot(overtime_df, ax[3], 'Elapsed Time', 'play_count', 'VPL')
    plt.tight_layout()
    ax[0].set_ylabel('Views')
    ax[1].set_xlabel('Minutes since publication')
    #print(f"Average VPL: {np.mean(overtime_df['VPL'])}")

def print_stats(df):        
    print(f"Number of videos: {len(df['VPE'])}")
    print(f"Average VPL: {np.mean(df['VPL'])}")
    print(f"Average Engagements: {np.mean(df['Engagements'])}")
    print(f"Average VPE: {np.mean(df['VPE'])}")
    print(f"Use hashtags: {np.mean(df['any_hashtags'])}")

def print_extra_stats(username):
    cur.execute('''
                select id,
                sum(case when hashtag_name is not null then 1 else 0 end) tags,
                sum(case when tagged_user is not null then 1 else 0 end) users,
                max(fetch_time - create_time) as "Elapsed Time"
                 from tiktok.videos_all_materialized m
                 left join tiktok.text_extra using (id)
                 where author = (%s)
                 group by id
                ''', [username])        
    df = pd.DataFrame(cur.fetchall(), 
                               columns = [desc[0] for desc in cur.description])   
    print(f"Average hashtags: {np.mean(df['tags'])}")
    print(f"Average tagged users: {np.mean(df['users'])}")

# =============================================================================
# 
# Configuration section
# 
# =============================================================================

username ='mathematicanese' 
load_only = True

# =============================================================================
# 
# Main stuff
# 
# =============================================================================
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

if load_only:
    load(username)
else:
    cur.execute('''SELECT *
    from tiktok.videos_materialized
    where author = (%s)''', [username])
    res=cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(res, columns = colnames)
    df['VPL'] = df['play_count'] / df['like_count']
    df['Engagements'] = np.sum(df[['like_count', 'share_count', 'comment_count']],
                               axis = 1)
    df['VPE'] = df['play_count'] / df['Engagements']
    
    histogram(df)
    plt.figure()
    view_like(df)
    most_recent(username)
    print_stats(df)
    print_extra_stats(username)

cur.close()
conn.close()
