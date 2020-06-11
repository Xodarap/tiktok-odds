# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:09:02 2020

@author: bwsit
"""

import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import powerlaw
from scipy import optimize
from scipy.stats import beta
from scipy.stats import powerlognorm
from scipy.stats import expon
from scipy.stats import geom
from scipy.stats import gamma
from scipy import stats
import pandas as pd

xmin, xmax = plt.xlim()

conn=psycopg2.connect('dbname=postgres user=postgres password=0FFzm4282FW^')
cur = conn.cursor()

cur.execute("""SELECT play_count, case when "first" is not null then 1 else 0 end "first"
            from tiktok.videos_materialized m
            left join (
                select min(create_time) first_video_time, author, 'first' "first",
                
                from tiktok.videos_materialized               
                group by author
            ) f
            on f.author = m.author and f.first_video_time = m.create_time s""")
res=cur.fetchall()
df = pd.DataFrame(columns = ['Views', 'First'])
df['Views'] = [np.log10(r[0]) if r[0] > 0 else 0 for r in res]
df['First'] = [r[1] for r in res]

first_views = df[df['First'] == 1]['Views']
later_views = df[df['First'] == 0]['Views']
n_bins=200
plt.hist(first_views, n_bins, facecolor='blue', alpha=0.5,density=True)
plt.hist(later_views, n_bins, facecolor='red', alpha=0.5,density=True)
print(f'First average: {np.mean(first_views)}. Later average: {np.mean(later_views)}')
print(stats.ttest_ind(first_views, later_views))

cur.close()
conn.close()