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
         6856842489123409157: 'p value',
         6857171668180143365: 'gpt3 algorithm advice',
         6857978583210544390: 'java c#',
         6857652821790149893: 'live redux',
         6858299206667422982: 'shadowban',
         6858349147012025605: 'auto r',
         6858696770914880774: 'python bare minimum',
         6859128521365736709: 'sigmoid',
         6859814030719110405: 'trends',
         6860178143399955718: 'trends - which',
         6860556782067092742: 'natalia'
         }

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""select all_data.id,
    all_data.play_count,
    all_data.share_count,
    all_data.comment_count,
    all_data.like_count,
    all_data.create_time,
    all_data.fetch_time,
    extract(epoch from (all_data.fetch_time - all_data.create_time)) / 60
   FROM ( SELECT (tiktok.json ->> 'id'::text)::bigint AS id,
            (tiktok.json -> 'author'::text) ->> 'uniqueId'::text AS author,
            ((tiktok.json -> 'stats'::text) -> 'playCount'::text)::integer AS play_count,
            ((tiktok.json -> 'stats'::text) -> 'shareCount'::text)::integer AS share_count,
            ((tiktok.json -> 'stats'::text) -> 'commentCount'::text)::integer AS comment_count,
            ((tiktok.json -> 'stats'::text) -> 'diggCount'::text)::integer AS like_count,
            to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision) AS create_time,
            tiktok."time" AS fetch_time,
            tiktok.representative
           FROM tiktok) all_data
		 where all_data.id in ( 
             --6852376935411010822, -- posting frequency
             --6851335018715811078 -- vpl
             --6838386972445248773 -- most popular (beautiful)
             --6845799211468918021 -- R
             --)
             --select 6853576412276788486
             --union all 
             --select 6838386972445248773
             --union all
             select id from (
                 select distinct (tiktok.json ->> 'id'::text)::bigint AS id,
                 to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision)
                 from tiktok
                 where (tiktok.json -> 'author'::text) ->> 'uniqueId'::text = 'benthamite'
                 order by to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision) desc
                 limit 3
             ) q
         ) 
          --union all (select 6852376935411010822, 0, 0, 0, 0, to_timestamp(0), to_timestamp(0), 0)
          order by fetch_time asc
""")

res=cur.fetchall()
result_df = pd.DataFrame(res, columns = ['ID', 'Views', 'Shares', 'Comments', 
                                         'Likes', 'Create Time', 'Fetch Time', 
                                         'Elapsed Time'])
# result_df = pd.concat([pd.DataFrame([[]], columns = ['ID', 'Views', 'Shares', 'Comments', 
#                                          'Likes', 'Create Time', 'Fetch Time', 
#                                          'Elapsed Time']), result_df])
result_df['VPL'] = result_df['Views'] / result_df['Likes']


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

for ident in np.unique(result_df['ID']):
    # run_subset(result_df, ident)
    pass

fig, axs = plt.subplots(2,1, figsize = (13, 8))
ax = axs[0]
result_df['Time in Seconds'] = [t.timestamp() for t in result_df['Fetch Time']]
ids = np.unique(result_df['ID'])
def plot_id(ids):
    one = result_df[result_df['ID'] == ids]
    label = id_map[ids] if ids in id_map else ''
    ax.plot(one['Elapsed Time'], one['Views'],
                  label = label)
    axs[1].plot(one['Elapsed Time'], one['Likes'],
                  label = label)
for idx in ids:
    plot_id(idx)
ax.set_ylabel('Views')
ax.set_xlabel('Minutes since publication')
vid_start = (pd.Timestamp('2020-08-10 20:52:00-07:00') - pd.Timedelta(hours = 1.4)) - pd.Timestamp('2020-08-07 13:57:31-07:00')
vid_start = vid_start.total_seconds() / 60
ax.plot([vid_start, vid_start], ax.get_ylim(), '--')
vid_end = (pd.Timestamp('2020-08-10 20:52:00-07:00')) - pd.Timestamp('2020-08-07 13:57:31-07:00')
vid_end  = vid_end.total_seconds() / 60
ax.plot([vid_end , vid_end ], ax.get_ylim(), '--')
ax.legend()
conn.close()