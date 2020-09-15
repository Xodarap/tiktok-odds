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
import os

folder = 'stream_4'

def histogram(df, username):
    plt.hist(np.log10(df['play_count']), density = True)
    plt.title(f'{username} play count distribution')
    plt.xlabel('log10(views)')
    plt.ylabel('Density')
    plt.savefig(f'D:/Documents/tiktok-live-graphs/{folder}/{username}_histogram.png')

def view_like(df, vpl, username):
    # plt.scatter(np.log10(df['play_count']), np.log10(df['like_count']))
    # xlim = plt.xlim()
    # x = np.linspace(np.power(10, xlim[0]), np.power(10, xlim[1]), 100)
    # plt.plot(x, np.log10(1 / (vpl / np.power(10, x))), color = 'r', 
    #          label = 'Your Average')
    # plt.plot(x, np.log10(1 / (4.556428 / np.power(10, x))), color = 'k', 
    #          label = '2020 Global Average')
    plt.scatter(df['play_count'], (df['like_count']))
    xlim = plt.xlim()
    x = np.linspace(xlim[0], xlim[1], 10)
    plt.plot(x, x / vpl, color = 'r', 
              label = 'Your Average')    
    y = x/4.556428
    plt.plot(x, y, color = 'k', 
              label = '2020 Global Average')
    plt.yscale('log', basey = 10)
    plt.xscale('log', basex = 10)
    plt.title(f'{username} plays vs. likes')
    plt.xlabel('log10(Plays)')
    plt.ylabel('log10(Likes)')
    plt.legend()
    plt.savefig(f'D:/Documents/tiktok-live-graphs/{folder}/{username}_view_like.png')

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
                 from tiktok.videos_normalized_all m
                where id = (
                    select id 
                    from tiktok.videos_normalized_all
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
    return np.mean(df['VPL'])

def print_extra_stats(username):
    cur.execute('''
                
select id,
sum(case when hashtag_name is not null then 1 else 0 end) tags,
sum(case when tagged_user is not null then 1 else 0 end) users
from (
select distinct id, hashtag_name, tagged_user
from
tiktok.text_extra 
where author = (%s) )q
group by id
                ''', [username])        
    df = pd.DataFrame(cur.fetchall(), 
                               columns = [desc[0] for desc in cur.description])   
    print(f"Average hashtags: {np.mean(df['tags'])}")
    print(f"Average tagged users: {np.mean(df['users'])}")
def get_tag(tag):
    cur.execute('select video_count, view_count from tiktok.tag_data where name = %s', (tag,))    
    existing = cur.fetchall()
    if len(existing) == 0:
        return None
    return existing[0]
def save_tag(tag, api):
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
    print(f'saved {tag}')
def get_or_save_tag(tag, api):
    try:
        if tag == '':
            return [0, 0]
        existing = get_tag(tag)
        if existing is not None:
            return existing
        save_tag(tag, api)
        return get_tag(tag)
    except:
        return [0, 0]
def print_tag_stats(username):
    api = TikTokApi()
    cur.execute('''
                select hashtag_name, count(1)
from (
	select distinct on (id, hashtag_name) hashtag_name
	from tiktok.text_extra 
	where author = (%s)
) q2
group by hashtag_name
order by count(1) desc 
limit 10
            ''', [username])        
    df = pd.DataFrame(cur.fetchall(), 
                               columns = [desc[0] for desc in cur.description])   
                              
    def vid_count(tag, api):
        try:
            return get_or_save_tag(tag, api)
            # res = api.getHashtagObject(tag)
            # return [res['challengeInfo']['stats']['videoCount'], res['challengeInfo']['stats']['viewCount']]
        
        except:
            return [0, 0]
    res = [get_or_save_tag(tag, api) for tag in df['hashtag_name']]
    df['video_count'] = [r[0] for r in res]
    df['video_count_human'] = ["{:,}".format(r[0]) for r in res]
    df['view_count'] = [r[1] for r in res]
    df['avg views'] = ["{:,.2f}".format(a) for a in df['view_count'] / df['video_count']]
    print('Ten most used tags:')
    print(df[['hashtag_name', 'count', 'video_count_human']])
    print('==========================')
    print(df[['hashtag_name', 'count', 'avg views']])
    cur.execute('''
                select hashtag_name, count(1)
                from (
        select distinct on (id, hashtag_name) hashtag_name
         from 
         tiktok.text_extra
         where author = (%s)
         and hashtag_name in (
             'fyp',
             'foryoupage',
             'xyzabc',
             'xyzbca',
             'viral',
             'foryou'
         )) q
         group by hashtag_name
         order by count(1) desc 
         limit 10
        ''', [username])        
    df = pd.DataFrame(cur.fetchall(), 
                               columns = [desc[0] for desc in cur.description])
    print('Any FYP tags:')
    print(df)

# =============================================================================
# 
# Configuration section
# 
# =============================================================================

username = os.environ.get('USER')

# =============================================================================
# 
# Main stuff
# 
# =============================================================================
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()


def load(username, disp = False):
    api = TikTokApi()
    if disp:
        print(f'Fetching {username}')
    try:
        tiktoks = api.byUsername(username, count=500)
    except Exception as e:
        print(e)
    t_time=str(datetime.datetime.now())
    if disp:
        print(len(tiktoks))
    for t in tiktoks:
        new_t=json.dumps(t)
        cur.execute('INSERT INTO tiktok (time,json) VALUES (%s,%s)', (t_time,new_t))
    conn.commit()
    if disp:
        print('Saved')

def run_user(username):
    print(f'========== {username} =========')
    load(username, True)
    cur.execute('''SELECT *
    from tiktok.videos_normalized
    where author = (%s)''', [username])
    res=cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(res, columns = colnames)
    df['VPL'] = df['play_count'] / df['like_count']
    df['Engagements'] = np.sum(df[['like_count', 'share_count', 'comment_count']],
                                axis = 1)
    df['VPE'] = df['play_count'] / df['Engagements']
    
    try:
        histogram(df, username)
    except:
        print('oops')
    plt.figure()
    vpl = print_stats(df)
    view_like(df, vpl, username)
    #most_recent(username)
    print_extra_stats(username)
    print_tag_stats(username)


names = [username]
for name in names: 
    run_user(name)
    # plt.figure()

cur.close()
conn.close()

plt.show()