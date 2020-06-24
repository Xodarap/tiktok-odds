# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:51:32 2020

@author: bwsit
"""
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

query = '''
SELECT "soundId", count(1), sum("playCount") plays, sum("diggCount") likes, 
sum("shareCount") shares, sum("commentCount") commentCount,
sum("diggCount") / sum("playCount") lpv
from tiktok.videos
	where "diggCount" < "playCount"
group by videos."soundId"
having count(1) > 500;
'''
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute(query)
res=cur.fetchall()
print('fetched')
result_df = pd.DataFrame(res, columns = ['Sound ID', 'Count', 'Views', 'Likes',
                                         'Shares', 'Comments', 'LPV'])
for column in result_df.columns:
    result_df[column] = pd.to_numeric(result_df[column])

fig, ax = plt.subplots(2,1)
ax[0].hist(result_df['LPV'].astype('float'), density = True, bins = np.linspace(0,1.5,20))
ax[1].scatter(result_df['LPV'], result_df['Count'])