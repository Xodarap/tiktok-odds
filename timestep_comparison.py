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
select vd.id, vd2.lps lps_prev, vd.pps, vd_hx.avg_ppl, vd2.pps pps_prev
from tiktok.videos_delta vd
inner join tiktok.videos_delta vd2 on vd.id = vd2.id and vd.rn = vd2.rn + 1
inner join (
select vd.id, vd.rn, avg(vd2.ppl::float) avg_ppl
from tiktok.videos_delta vd
inner join tiktok.videos_delta vd2 on vd2.rn <= vd.rn
and vd.id = vd2.id
group by vd.id, vd.rn) vd_hx on vd_hx.id = vd.id and vd.rn = vd_hx.rn
where vd.d_play > 0 and vd.d_time between '8 minutes' and '12 minutes'
""")

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
fig, ax = plt.subplots(2,1,sharex = False)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
ax[0].scatter(df['lps_prev'] * df['avg_ppl'], df['pps'], s = 0.5)
x = np.linspace(0, 5, 100)
ax[0].plot(x, x ,'r-')
ax[0].set_xlim(0, 5)
ax[0].set_ylim(0, 5)
ax[1].scatter(df['pps_prev'], df['pps'], s = 0.5)
ax[1].plot(x, x,'r-')