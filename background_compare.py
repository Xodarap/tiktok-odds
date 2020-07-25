# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:24:03 2020

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
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
r_vid = 6852704962686766341
cur.execute('''
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select sum(pps) sum_pps,
            sum(lps) sum_lps,
            r_vid,
            rounded_time,
            count(1) cnt
            from (select id, pps, lps,
            case when d.id = (%s) then 1 else 0 end r_vid,
            to_timestamp(floor(extract('epoch' from fetch_time_2) / 600) * 600) rounded_time
             from tiktok.videos_delta d
             inner join tiktok.videos_normalized using(id)
             where d.author = (%s)
             and fetch_time_2 > '2020-07-22'
             ) q
             where id in (
                 select distinct id 
                 from tiktok.videos_normalized
                 where create_time  <= 
                     (select create_time from tiktok.videos_normalized where id = (%s) limit 1)
             )
             and
             id in (
                 select distinct id 
                     from tiktok.videos_delta
                     where fetch_time_2  >= '2020-07-23 09:00:00'
             )
             group by r_vid, rounded_time
            ''', [r_vid, 'benthamite', r_vid])        
df = pd.DataFrame(cur.fetchall(), 
                           columns = [desc[0] for desc in cur.description])   
conn.commit()
cur.close()
conn.close()

fig, ax = plt.subplots(2,1,sharex = True)
r_vid = df[df['r_vid'] == 1]
other = df[df['r_vid'] == 0]
ax[0].plot_date(r_vid['rounded_time'], r_vid['sum_pps'])
ax[1].plot_date(other['rounded_time'], other['sum_pps'])
fig.autofmt_xdate()