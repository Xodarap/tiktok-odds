import datetime
import time
import psycopg2
import json
from random import randint
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
load_dotenv()
from scipy import stats

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()


cutoffs = {'mattupham': '2020-08-17', 
           'bri_xu': '2020-08-18', 
           'amarchenkova': '2020-08-18', 
           '3dprintingguru': '2020-05-20',
         'dilmerval': '2020-08-31', 
         }

non_fund = ['dailyalgo', 'benthamite', 'shante.tech']
for user in non_fund:
    cutoffs[user] = '2020-08-17'

cur.execute("""
        SELECT author, create_time, play_count 
        FROM tiktok_normalized
        where author = ANY(%s)
        """, (list(cutoffs.keys()),))
res = cur.fetchall()        
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
df['create_time_dt'] = pd.to_datetime(df['create_time'], utc = True)
for user, dt in cutoffs.items():
    ndt = pd.to_datetime(dt, utc = True)
    df.loc[df['author'] == user, 'before'] = df.loc[df['author'] == user, 'create_time_dt'] < ndt

def analyze_subset(df, ax, title):
    df['before'] = df['before'].astype('boolean')
    before_views = df.loc[df['before'], 'play_count']
    after_views = df.loc[~df['before'], 'play_count']

    ax.violinplot([before_views.values, after_views.values])
    ax.set_yscale('log')
    ax.set_xticklabels(['', 'before', '', 'after'])
    ax.set_title(title)
    
    _, p = stats.ks_2samp(before_views, after_views)
    print(f"KS for {title}: {p}")
    print(f"Mean for {title}: {np.mean(before_views)}, {np.mean(after_views)}")
    print(f"Median for {title}: {np.median(before_views)}, {np.median(after_views)}")
    print(f"Variance for {title}: {np.var(before_views)}, {np.var(after_views)}")
    print(f"Number of videos for {title}: {len(before_views)}, {len(after_views)}")
    return [title, np.mean(before_views), np.mean(after_views),
            np.median(before_views), np.median(after_views),
            p]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
analyze_subset(df, ax[0,0], 'All Creators')
analyze_subset(df[df['author'].isin(non_fund)], ax[0,1], 'Non Fund Creators')
analyze_subset(df[~df['author'].isin(non_fund)], ax[1,0], 'Fund Creators')

relevant = cutoffs.keys() #['bri_xu']
fig, axs = plt.subplots(nrows=len(relevant), figsize=(6, 6))
results = []
for username,ax in zip((relevant), axs):
    results.append(analyze_subset(df[df['author'] == username], ax, username))

result_df = pd.DataFrame(results, columns = ['title', 'meanb', 'meana', 'medb', 'meda', 'p'])
# pairs = [get_videos(u, c) for u, c in cutoffs.items()]

# def 