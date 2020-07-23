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

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select vd.*, avgs.avg_ppl, avg_pps, vd2.ppl next_ppl, vd2.pps next_pps
            from tiktok.videos_delta vd
            inner join 
            (select id, avg(ppl) avg_ppl,
            avg(pps) avg_pps
            from tiktok.videos_delta
            group by id) avgs using (id)
            inner join tiktok.videos_delta vd2 on vd.id = vd2.id and
            vd.rn = vd2.rn - 1
            where vd.id in 
            (select distinct id
            from tiktok.videos_delta
            where d_play > 0
            and author = 'benthamite'
            group by id
            having count(1) >10
            )
            order by vd.id, elapsed_seconds_2 asc
""")

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
fig, ax = plt.subplots(2,1,sharex = False)
# df['prev_pps'] = df['pps'].shift(1)
# df['prev_ppl'] = df['ppl'].shift(1)
# pps = df['pps']
# df = df[pps.between(pps.quantile(0.05), pps.quantile(0.95))]
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].scatter(df['next_ppl'], df['ppl'])
ax[0].set_ylim(0.001, 5000)
ax[0].set_xlim(0.9, 150)

ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].scatter(df['ppl'] / df['avg_ppl'], df['next_pps'] / df['avg_pps'])
ax[1].set_ylim(0.001, 5000)
ax[1].set_xlim(0.001, 5000)