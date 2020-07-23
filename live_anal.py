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

cur.execute("""
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select fetch_time_2, sum(pps) pps, count(1) number_of_videos,
            sum(lps) lps, avg(ppl) vpl
            from tiktok.videos_delta
            inner join tiktok.videos_materialized m using (id)
            where m.author = 'benthamite' and
            fetch_time_2 between '2020-07-11 17:00:00-07' and  '2020-07-11 23:00:00-07'
            group by fetch_time_2
            order by fetch_time_2 asc
""")

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
fig, ax = plt.subplots(1,1,sharex = True)
ax = [ax]

ax[0].plot_date(df['fetch_time_2'], df['pps'], fmt = 'x', label = 'Plays per second')
start = np.datetime64('2020-07-11 19:30:00-07')
end = np.datetime64('2020-07-11 20:53:00-07')
ax[0].set_ylim(0, ax[0].get_ylim()[1])
ax[0].plot_date([start, start], ax[0].get_ylim(), fmt = '--', label = 'Live Start')
ax[0].plot_date([end, end], ax[0].get_ylim(), fmt = '--', label = 'Live End')
ax[0].legend()
ax[0].set_ylabel('Plays per second')
# ax[1].plot_date(df['fetch_time_2'], df['lps'], label = 'Likes per second')
# ax[1].plot_date([start, start], ax[1].get_ylim(), fmt = '-', label = 'Live Start')
# ax[1].plot_date([end, end], ax[1].get_ylim(), fmt = '-', label = 'Live End')
# ax[1].legend()
fig.autofmt_xdate()

df['fetch_localized'] = [v.to_numpy() for v in df['fetch_time_2']]
before = df.loc[df['fetch_localized'] < start, 'pps']
after = df.loc[(df['fetch_localized'] >= start) & (df['fetch_localized'] <= end), 'pps']
results = stats.ks_2samp(before, after)
print(f'KS 2-sample p-value: {results[1]}')
print(f'Ratio: {np.mean(after)/np.mean(before)}')