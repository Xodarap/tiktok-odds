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
		 where all_data.id in ( --6852376935411010822, -- posting frequency
         6851335018715811078 -- vpl
         --6838386972445248773 -- most popular (beautiful)
         --6845799211468918021 -- R
         )
         /*select id from (
             select distinct (tiktok.json ->> 'id'::text)::bigint AS id,
             to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision)
             from tiktok
             where (tiktok.json -> 'author'::text) ->> 'uniqueId'::text = 'benthamite'
             order by to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision) desc
             limit 2
             ) q
         ) */
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

def ev_binomial(a, b, views, likes):
    a_new = a + likes
    b_new = b + views - likes
    return a_new / (a_new + b_new)

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


def run_subset(df, ident):
    result_df = df[df['ID'] == ident]    
    result_df['Views_1'] = result_df['Views'].shift(1)
    result_df['Likes_1'] = result_df['Likes'].shift(1)
    result_df['d_v'] = (result_df['Views'] - result_df['Views_1']).rolling(window = 1).mean()
    result_df['d_v'] = interpolate_zeros(result_df['d_v'])
    result_df['d_l'] = (result_df['Likes'] - result_df['Likes_1']).rolling(window = 1).mean()
    result_df['d_l_1'] = result_df['d_l'].shift(1)
    #result_df['d_vpl'] = (result_df['d_v']) / (result_df['d_l_1'])
    result_df['d_vpl'] = 1 / ev_binomial(10, 50, result_df['d_v'], result_df['d_l_1'])
    result_df['d_vpl_rolling'] = result_df['d_vpl'].rolling(window = 3).mean()
    result_df['inferred_vpl'] = np.cumsum(result_df['d_v']) / np.cumsum(result_df['d_l'])
    fig, ax = plt.subplots(2,1, sharex = True)
    ax[0].plot(result_df['Elapsed Time'], result_df['Views'])
    ax[1].plot(result_df['Elapsed Time'], result_df['d_vpl_rolling'])
    plt.tight_layout()
    ax[0].set_ylabel('Views') 
    ax[1].set_xlabel('Minutes since publication')
    ax[1].set_ylabel('Views Per Like')

for ident in np.unique(result_df['ID']):
    run_subset(result_df, ident)