t# -*- coding: utf-8 -*-
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
            REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select id, pps, ppl, ppe, play_count_2, elapsed_seconds_2,
            like_count_2
            from tiktok.videos_delta
            where id in 
            (select distinct id
            from tiktok.videos_delta
            where d_play > 0
            group by id
            having count(1) > 20)
            --and id in (6838386972445248773, 6844191231224876293)
            order by id, elapsed_seconds_2 asc
""")

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = ['id', 'pps', 'ppl', 'ppe', 'plays', 'elapsed_seconds', 'like_count_2'])
fig, ax = plt.subplots(3,1,sharex = False)
for num, idx in enumerate(np.unique(df['id'])):
    video = df[df['id'] == idx]
    color = None #list(mcolors.CSS4_COLORS.keys())[num]
    label = None
    marker = None
    if idx == 6844191231224876293:
        label = 'Recent'
        color = 'r'
        marker = '.'
    ax[0].plot(video['elapsed_seconds'], video['plays'], color = color, 
             label = label, marker = marker)
    ax[1].plot(video['elapsed_seconds'], video['plays'] / video['like_count_2'], color = color, 
             label = label, marker = marker)
    ax[2].plot(video['elapsed_seconds'], video['ppe'], color = color, 
             label = label, marker = marker)
#plt.legend()
# ax[0].set_xlim(0,186400)
# ax[1].set_xlim(0,186400)
# ax[1].set_ylim(0,10)
#plt.scatter(np.log(df['pps']), df['ppe'])
