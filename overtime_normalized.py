# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:20:42 2020

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


id_map = {6852376935411010822: 'posting frequency',
         6851335018715811078: 'vpl',
         6838386972445248773: 'most popular (beautiful)',
         6845799211468918021: 'R',
         6855359832392731909: 'sound delay',
         6855018062626753798: 'spaghetti code',
         6854631030050000133: 'unschooling',
         6853576412276788486: 'bruh girls',
         6854222855345835270: 'tabs vs spaces',
         6855694311799999749: 'original vs new sounds',
         6855891275665706246: 'kolmogorov smirnov',
         6856080442144099590: 'duration',
         6856426933966621958: 'valgrind',
         6856492989137603846: 'gpt3 mean girls',
         6856842489123409157: 'p value'
         }

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""            
select id,
play_count,
share_count,
like_count,
extract(epoch from (fetch_time - create_time)) / 60 elapsed_time,
fetch_time
from videos_normalized_all
where author = 'benthamite'
and id in (
    select distinct id 
    from videos_normalized_all
    where extract(epoch from (fetch_time - create_time)) / 60 > 1440
)
and id in (
    select distinct id 
    from videos_normalized_all
    where extract(epoch from (fetch_time - create_time)) / 60 < 1440
    group by id
    having count(1) > 20
    limit 10
)
and id in (
    select distinct id 
    from videos_normalized_all
    where extract(epoch from (fetch_time - create_time)) / 60 < 60
)
and id in (
    select distinct id 
    from videos_normalized_all
    where extract(epoch from (fetch_time - create_time)) / 60 > 1300
)
and extract(epoch from (fetch_time - create_time)) / 60 < 1440
order by fetch_time asc
""")

res=cur.fetchall()
result_df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
result_df['VPL'] = result_df['play_count'] / result_df['like_count']


# plt.plot(result_df['VPL'], result_df['inferred_vpl'])                    
def interpolate_zeros(vals):
    out = vals.copy().reindex(range(0, len(vals)))
    start_idx = None
    def interpolate(vals, start, end):
        buckets = (end+1)-start
        amount = vals[end]
        vals[start:(end + 1)] = amount / buckets
    for idx in range(0, len(vals)):
        if start_idx is not None:
            if out[idx] > 0:
                interpolate(out, start_idx, idx)
                start_idx = None
        elif out[idx] == 0:
            start_idx = idx
    return out

def make_plot(df, ax, x, y, y2):
    l1, = ax.plot(df[x], df[y], 'b-', label = y)
    ax2 = ax.twinx()
    l2, = ax2.plot(df[x], df[y2], 'r-', label = y2)
    ax.set_title(y2)
    ax.legend(handles = [l1, l2])

def run_subset(df, ident):
    result_df = df[df['ID'] == ident]    
    result_df['Views_1'] = result_df['Views'].shift(1)
    result_df['Likes_1'] = result_df['Likes'].shift(1)
    result_df['d_v'] = (result_df['Views'] - result_df['Views_1']).rolling(window = 3).mean()
    result_df['d_v'] = interpolate_zeros(result_df['d_v'])
    result_df['d_l'] = (result_df['Likes'] - result_df['Likes_1']).rolling(window = 3).mean()
    result_df['d_l_1'] = result_df['d_l'].shift(1)
    result_df['d_vpl'] = (result_df['d_v']) / (result_df['d_l_1'])
    result_df['d_vpl_rolling'] = result_df['d_vpl']
    result_df['inferred_vpl'] = np.cumsum(result_df['d_v']) / np.cumsum(result_df['d_l'])
    #plt.plot(result_df['Elapsed Time'], result_df['Views'])
    #plt.ylabel('Views')
    #plt.xlabel('Seconds since publication')
    fig, ax = plt.subplots(1,4, figsize = (13, 8))
    make_plot(result_df, ax[0], 'Elapsed Time', 'Views', 'VPL')
    make_plot(result_df, ax[1], 'Elapsed Time', 'Views', 'd_v')
    make_plot(result_df, ax[2], 'Elapsed Time', 'Views', 'd_l')
    make_plot(result_df, ax[3], 'Elapsed Time', 'Views', 'd_vpl_rolling')
    plt.tight_layout()
    ax[0].set_ylabel('Views') 
    ax[1].set_xlabel('Minutes since publication')


fig, ax = plt.subplots(1,1, figsize = (13, 8))
result_df['Time in Seconds'] = [t.timestamp() for t in result_df['fetch_time']]
ids = np.unique(result_df['id'])
def plot_id(ids):
    one = result_df[result_df['id'] == ids]
    one['normalized_views'] = one['play_count'] / max(one['play_count'])
    ax.plot(one['elapsed_time'], one['normalized_views'])
                  # label = id_map[ids])
for idx in ids:
    plot_id(idx)
ax.set_ylabel('Views')
ax.set_xlabel('Minutes since publication')
ax.legend()