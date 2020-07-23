# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:09:56 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.special import gammaln
from scipy import stats
import pandas as pd
from collections.abc import Iterable
from scipy.stats import wilcoxon
import matplotlib.colors as mcolors

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select id, pps, ppl, ppe, play_count_2, elapsed_seconds_2,
            like_count_2, d_play, d_like
            from tiktok.videos_delta
            where /*id in 
            
            (select distinct id
            from tiktok.videos_delta
            where d_play > 0
            group by id
            --having count(1) > 20
            )
            
            and */id in  (6852376935411010822, --freq
            6851335018715811078) --vpl
            --(6838386972445248773, 6844191231224876293)
            order by id, elapsed_seconds_2 asc
""")

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
df['rolling_ppl'] = df['ppl'].rolling(window = 1).mean()
df.loc[df['rolling_ppl'] > 8, 'rolling_ppl'] = 8
df['elapsed_minutes'] = df['elapsed_seconds_2'] / 60
fig, ax = plt.subplots(3,1,sharex = False)

def interpolate_zeros(vals):
    out = vals.copy()
    start_idx = None
    def interpolate(vals, start, end):
        buckets = (end+1)-start
        amount = vals[end]
        vals[start:(end + 1)] = amount / buckets
    for idx in range(0, len(vals)):
        if start_idx is not None:
            if vals[idx] > 0:
                interpolate(out, start_idx, idx)
                start_idx = None
        elif vals[idx] == 0:
            start_idx = idx
    return out

df['d_plays_interp'] = interpolate_zeros(df['d_play'])
df['ppl_interp'] = df['d_plays_interp'] / df['d_like']
# df['d_likes_interp'] = interpolate_zeros(df['d_like'])

#730 pm start live
for num, idx in enumerate(np.unique(df['id'])):
    video = df[df['id'] == idx]
    color = None #list(mcolors.CSS4_COLORS.keys())[num]
    label = None
    marker = None
    if idx == 6844191231224876293:
        label = 'Recent'
        color = 'r'
        marker = '.'
    ax[0].plot(video['elapsed_minutes'], video['play_count_2'], color = color, 
             label = label, marker = marker)
    ax[0].set_ylabel('Plays')
    ax[1].plot(video['elapsed_minutes'], video['ppl_interp'], color = color, 
             label = label, marker = marker)
    ax[1].set_ylabel('Rolling ppl')
    ax[2].plot(video['elapsed_minutes'], video['pps'], color = color, 
             label = label, marker = marker)
    ax[2].set_ylabel('pps')
    ax[2].plot([92420, 92420], ax[2].get_ylim())
    ax[2].plot([96020, 96020], ax[2].get_ylim())
    for i in range(0,3):
        ax[i].set_xlim(000, 2000)
#plt.legend()
# ax[0].set_xlim(0,186400)
# ax[1].set_xlim(0,186400)
# ax[1].set_ylim(0,10)
#plt.scatter(np.log(df['pps']), df['ppe'])
