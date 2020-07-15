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
		 where all_data.id in ( --6848018597407771909
         select id from (
             select distinct (tiktok.json ->> 'id'::text)::bigint AS id,
             to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision)
             from tiktok
             where (tiktok.json -> 'author'::text) ->> 'uniqueId'::text = 'benthamite'
             order by to_timestamp((tiktok.json -> 'createTime'::text)::integer::double precision) desc
             limit 2
             ) q
         )
          order by fetch_time asc
""")

res=cur.fetchall()
result_df = pd.DataFrame(res, columns = ['ID', 'Views', 'Shares', 'Comments', 'Likes', 'Create Time', 'Fetch Time', 'Elapsed Time'])
result_df['VPL'] = result_df['Views'] / result_df['Likes']
                    

def make_plot(df, ax, x, y, y2):
    l1, = ax.plot(df[x], df[y], 'b-', label = y)
    ax2 = ax.twinx()
    l2, = ax2.plot(df[x], df[y2], 'r-', label = y2)
    ax.set_title(y2)
    ax.legend(handles = [l1, l2])

def run_subset(df, ident):
    result_df = df[df['ID'] == ident]
    #plt.plot(result_df['Elapsed Time'], result_df['Views'])
    #plt.ylabel('Views')
    #plt.xlabel('Seconds since publication')
    fig, ax = plt.subplots(1,4, sharey = 'row', figsize = (13, 8))
    make_plot(result_df, ax[0], 'Elapsed Time', 'Views', 'Likes')
    make_plot(result_df, ax[1], 'Elapsed Time', 'Views', 'Shares')
    make_plot(result_df, ax[2], 'Elapsed Time', 'Views', 'Comments')
    make_plot(result_df, ax[3], 'Elapsed Time', 'Views', 'VPL')
    plt.tight_layout()
    ax[0].set_ylabel('Views')
    ax[1].set_xlabel('Minutes since publication')

for ident in np.unique(result_df['ID']):
    run_subset(result_df, ident)

fig, ax = plt.subplots(1,1, figsize = (13, 8))
result_df['Time in Seconds'] = [t.timestamp() for t in result_df['Fetch Time']]
ids = np.unique(result_df['ID'])
one = result_df[result_df['ID'] == ids[0]]
two = result_df[result_df['ID'] == ids[1]]
l1, = ax.plot(one['Elapsed Time'], one['Views'], label = 'Video 1')
l2, = ax.plot(two['Elapsed Time'], two['Views'], label = 'Video 2')
ax.set_ylabel('Views')
ax.set_xlabel('Minutes since publication')
ax.legend(handles = [l1, l2])