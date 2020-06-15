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


conn=psycopg2.connect('dbname=postgres user=postgres password=0FFzm4282FW^')
cur = conn.cursor()

cur.execute("""select
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
     JOIN ( SELECT (tiktok.json ->> 'id'::text)::bigint AS id,
            max(tiktok."time") AS latest_fetch
           FROM tiktok
          GROUP BY ((tiktok.json ->> 'id'::text)::bigint)) latest ON latest.id = all_data.id 
		  --AND latest.latest_fetch = all_data.fetch_time
     LEFT JOIN ( SELECT text_extra.id,
            bool_or(text_extra.hashtag_name <> ''::text) AS any_hashtags,
            bool_or(text_extra.tagged_user IS NOT NULL) AS any_tagged_users,
            bool_or(text_extra.is_commerce) AS is_commerce
           FROM tiktok.text_extra
          GROUP BY text_extra.id) te ON te.id = all_data.id
		  --where all_data.id = 6837152281259969797
        where all_data.id = 6838386972445248773
          order by fetch_time asc
""")

res=cur.fetchall()
result_df = pd.DataFrame(res, columns = ['Views', 'Shares', 'Comments', 'Likes', 'Create Time', 'Fetch Time', 'Elapsed Time'])

                     
fig, ax = plt.subplots(1,3, sharey = 'row')

def make_plot(df, ax, x, y, y2):
    ax.plot(df[x], df[y], 'b-')
    ax2 = ax.twinx()
    ax2.plot(df[x], df[y2], 'r-')
    ax.set_title(y2)

make_plot(result_df, ax[0], 'Elapsed Time', 'Views', 'Likes')
make_plot(result_df, ax[1], 'Elapsed Time', 'Views', 'Shares')
make_plot(result_df, ax[2], 'Elapsed Time', 'Views', 'Comments')
#fig.
plt.tight_layout()