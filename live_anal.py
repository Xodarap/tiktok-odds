# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 09:40:41 2020

@author: bwsit
"""


# -*- coding: utf-8 -*-
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
from datetime import datetime

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
def fetch_time(start, end, ax):
    cur.execute("""
                --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
                select fetch_time_2, sum(pps) pps, count(1) number_of_videos,
                sum(lps) lps, avg(ppl) vpl
                from tiktok.videos_delta
                inner join tiktok.videos_materialized m using (id)
                where m.author = 'benthamite' and
                fetch_time_2 between %s and %s
                group by fetch_time_2
                order by fetch_time_2 asc
    """, ((start - pd.Timedelta(hours = 3)), (end  + pd.Timedelta(hours = 3))))
    
    res=cur.fetchall()
    df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
    
    df['pps_norm'] = df['pps'] / df['number_of_videos']
    ax.plot_date(df['fetch_time_2'], df['pps_norm'], fmt = '-', label = 'Plays per second')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.plot_date([start, start], ax.get_ylim(), fmt = '--', label = 'Live Start')
    ax.plot_date([end, end], ax.get_ylim(), fmt = '--', label = 'Live End')
    # ax2 = ax.twinx()
    # ax2.plot_date(df['fetch_time_2'], df['number_of_videos'])
    # ax[0].legend()
    ax.set_ylabel('Plays per second')
    # ax[1].plot_date(df['fetch_time_2'], df['lps'], label = 'Likes per second')
    # ax[1].plot_date([start, start], ax[1].get_ylim(), fmt = '-', label = 'Live Start')
    # ax[1].plot_date([end, end], ax[1].get_ylim(), fmt = '-', label = 'Live End')
    # ax[1].legend()
    fig.autofmt_xdate()
    
    df['fetch_localized'] = [v.to_numpy() for v in df['fetch_time_2']]
    inside_rows = (df['fetch_time_2'] >= start) & (df['fetch_time_2'] <= end)
    outside = df.loc[~inside_rows , 'pps_norm']
    inside = df.loc[inside_rows, 'pps_norm']
    results = stats.ks_2samp(outside, inside)
    print(f'KS 2-sample p-value: {results[1]}')
    print(f'Ratio: {np.mean(outside)/np.mean(inside)}')
    return outside, inside

ranges = [
    [pd.Timestamp('2020-07-11 19:30:00-07'), pd.Timestamp('2020-07-11 20:53:00-07')],
    [pd.Timestamp('2020-07-22 20:30:00-07'), pd.Timestamp('2020-07-22 21:53:00-07')],
    [pd.Timestamp('2020-07-25 21:16:00-07'), pd.Timestamp('2020-07-25 22:24:00-07')],    
    [pd.Timestamp('2020-07-28 15:50:00-07'), pd.Timestamp('2020-07-28 17:00:00-07')],    
    [pd.Timestamp('2020-07-30 20:00:00-07'), pd.Timestamp('2020-07-30 21:30:00-07')],
    [pd.Timestamp('2020-08-01 11:11:00-07'), pd.Timestamp('2020-08-01 13:05:00-07')]
    ]
def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax

fig, axs = make_subplots(len(ranges),1, figsize = (9,12))
outsides = []
insides = []
for rng, ax in zip(ranges,axs):
    outs, ins = fetch_time(*rng, ax)
    outsides.extend(list(outs))
    insides.extend(list(ins))
axs[0].legend()
results = stats.ks_2samp(outsides, insides)
print(f'===Overall===')
print(f'KS 2-sample p-value: {results[1]}')
print(f'Ratio: {np.mean(outsides)/np.mean(insides)}')
conn.commit()
conn.close()