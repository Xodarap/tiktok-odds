# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:37:40 2020

@author: bwsit
"""
import psycopg2
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""
    SELECT author, max(follower_count)
    from tiktok.users_normalized u
    inner join tiktok.videos_materialized v using (author)
    where date_part('year', v.create_time) = (2020)
    and v.representative
    group by author
            """)
all_results = cur.fetchall()
conn.close()            

df = pd.DataFrame(all_results, columns = {'Author': 'string', 
                                          'Followers': 'int'})
df['Log Followers'] = np.log(df['Followers'], where = df['Followers'] > 0)

plt.hist(df['Log Followers'], density = True, bins = 20)

sorted_values = df['Followers'].sort_values(ignore_index = True)
entries_per_percentile = int(len(sorted_values)/100)
percentiles = [(percentile, sorted_values[percentile * entries_per_percentile]) for percentile in range(1,100)]