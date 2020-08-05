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
from scipy.optimize import curve_fit

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
id_map = {6852376935411010822: 'posting frequency',
         6851335018715811078: 'vpl',
         6838386972445248773: 'most popular (beautiful)',
         6845799211468918021: 'R',
         6855359832392731909: 'sound delay',
         6855018062626753798: 'spaghetti code',
         6854631030050000133: 'unschooling',
         6853576412276788486: 'bruh girls',
         6854222855345835270: 'tabs vs spaces',
         6855694311799999749: 'original vs new sounds',
         6855891275665706246: 'kolmogorov smirnov',
         6856080442144099590: 'duration',
         6856426933966621958: 'valgrind',
         6856492989137603846: 'gpt3 mean girls'
         }
ids = (6852376935411010822,
  6851335018715811078,
  6838386972445248773,
  6845799211468918021,
  6855359832392731909,
  6855018062626753798,
  6854631030050000133,
  6853576412276788486,
  6854222855345835270,
  6855694311799999749,
  6855891275665706246,
  6856080442144099590,
  6856426933966621958,
  6856492989137603846)
sql = """
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
select vd.id, vd2.lps lps_prev, vd.pps, vd_hx.avg_ppl, extract('seconds' from vd.d_time) d_time,
vd2.pps pps_prev, vd2.play_count_2 play_count_prev,
vd.rn, vd.elapsed_seconds_2, vd.play_count_2
from tiktok.videos_delta vd
inner join tiktok.videos_delta vd2 on vd.id = vd2.id and vd.rn = vd2.rn + 1
inner join (
select vd.id, vd.rn, avg(vd2.ppl::float) avg_ppl
from tiktok.videos_delta vd
inner join tiktok.videos_delta vd2 on vd2.rn <= vd.rn
and vd.id = vd2.id
group by vd.id, vd.rn) vd_hx on vd_hx.id = vd.id and vd.rn = vd_hx.rn
where vd.d_play > 0 and vd.d_time between '8 minutes' and '12 minutes'
and vd.elapsed_seconds_2 <= 43200 
and vd.id in (
6856426933966621958,
6855891275665706246
,6853576412276788486

)
--and vd.id = 6851335018715811078
order by id, rn asc
--limit 100
"""
# statement = cur.mogrify(sql, [ids])
# print(cur.mogrify(sql, [ids]))
cur.execute(sql,  
)

res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])

def sigmoid_kgrow(x, k, x0, scale, kgrow):
    return (1 + kgrow * x) * scale * 1.0 / (1 + np.exp(-k * (x - x0)))
def sigmoid(x, k, x0, scale):
    return scale * 1.0 / (1 + np.exp(-k * (x - x0)))
def gompertz(x, a, b, c, d):
    return a * np.exp(-b * np.exp(-(x- d)/c))
def verhulst(x, k, r, scale):
    return scale * 1.0 / (1 + k * np.exp(-r * x))
def log_gompertz(x, a, b, c):
    d = 0
    return np.log(a) + (-b * np.exp(-(x- d)/c))
def log_fit(x, y, fn, x0, maxfev):
    return curve_fit(fn, x, np.log(y), x0, maxfev = maxfev)
def fit_hx(df, x0, rn, fn):
    hx = df[df['rn'] < rn]
    if len(hx) < 5:
        return pd.Series([0])
    try:
        popt, pcov = curve_fit(fn, hx['elapsed_seconds_2'], hx['play_count_2'],
                           x0,
                           maxfev = 10000)
        predicted_next = fn(df.loc[df['rn'] == rn, 'elapsed_seconds_2'], *popt)
        return predicted_next
    except:
        return pd.Series([0])
    
def fit_wrapped(fn, relevant, x0, idx):    
    return relevant.apply(lambda r: fit_hx(relevant, x0, r['rn'], fn).values[0], 
                                           axis = 1)
def make_wrapped(fn, x0):
    def temp(relevant, idx):
        return fit_wrapped(fn, relevant, x0, idx)
    return temp

def fit_naive(df, idx):
    return df['play_count_prev'] + df['d_time'] * df['pps_prev']

def fit_lps(df, idx):
    return df['play_count_prev'] + df['d_time'] * df['lps_prev'] * df['avg_ppl']

def calc_error(relevant, col):
    predictable = relevant[relevant[col] > 0]
    return ((predictable['play_count_2'] - predictable[col])**2).sum()

predict_map = {#'naive': fit_naive,
               # 'lps': fit_lps,
                # 'sigmoid': make_wrapped(sigmoid, [1e-3, 1800, 1000])
#                'gompertz': make_wrapped(gompertz, [1.233985478768753637e+03
# -6.937349088651090234e+00
# -2.388625010146093344e-04
# -3.168001328951865503e+02]
# ),
#                 'verhulst': make_wrapped(verhulst, [2.899786421824289137e+05,
# 6.655495237487482358e-04,
# 1.067793852683489968e+04
# ])
               }

def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax

def full_sigmoid(relevant, idx):
    # ax.plot(relevant['elapsed_seconds_2'], relevant['play_count_2'], label = 'actual')
    x0 = [1e-3, 1800, 1e3, 1e-6]
    fn = sigmoid_kgrow
    # x0 = [2.899786421824289137e+05, 6.655495237487482358e-04, 1.067793852683489968e+04]
    popt, pcov = curve_fit(fn, relevant['elapsed_seconds_2'], relevant['play_count_2'],
                        x0,
                        maxfev = 10000)
    x = np.linspace(1, relevant['elapsed_seconds_2'].max(), 100)
    # popt = x0
    # scale = popt[2] + 1e-2 *x
    # sig = fn(x, popt[0], popt[1], scale)
    sig = fn(x, *popt)
    ax.plot(x, sig, label = 'sigmoid')

ids = pd.unique(df['id'])
fig, axs = make_subplots(len(ids), 1, figsize = (6, 13))
errs = []
for idx, ax in zip(ids, axs):
    relevant = df[df['id'] == idx]
    ax.plot(relevant['elapsed_seconds_2'], relevant['play_count_2'], label = 'actual')
    for col, fn in predict_map.items():
        relevant[col] = fn(relevant, idx)
        ax.plot(relevant['elapsed_seconds_2'], relevant[col], 'x-', label = col)
    full_sigmoid(relevant, idx)
    ax.set_title(id_map[idx])
    ax.set_ylabel('Views')
    ax.legend()
    errs.append([idx, id_map[idx],
                  *[calc_error(relevant, col) for col in predict_map.keys()]
                  ])
axs[-1].set_xlabel('Time after publication (seconds)')
err_df = pd.DataFrame(errs, columns = ['id', 'title', *predict_map.keys()])
total_err = pd.DataFrame(
        zip(predict_map.keys(),
            [err_df[col].sum() for col in predict_map.keys()]),
        columns = ['Method', 'Total Error']
    )
total_err['relative'] = total_err['Total Error'] / \
    total_err.loc[total_err['Method'] == 'naive', 'Total Error']
# hx = df
# plt.plot(hx['elapsed_seconds_2'], hx['play_count_2'])
# # popt, pcov = log_fit(hx['elapsed_seconds_2'], hx['play_count_2'], log_gompertz, 
# #                       [3000, 1, 2e4],
# #                         maxfev = 10000)
# popt, pcov = curve_fit(verhulst, hx['elapsed_seconds_2'], hx['play_count_2'],
#                         [1e5, 1e-3,10000],
#                         maxfev = 10000)
# plt.plot(hx['elapsed_seconds_2'], verhulst(hx['elapsed_seconds_2'], *popt))

# plt.plot(*zip(*bp))
# plt.plot(hx['elapsed_seconds_2'], sigmoid(hx['elapsed_seconds_2'], 1e-3, 1900, 1000))

# fig, ax = plt.subplots(2,1,sharex = False)
# # ax[0].set_yscale('log')
# # ax[0].set_xscale('log')
# ax[0].scatter(df['lps_prev'] * df['avg_ppl'], df['pps'], s = 0.5)
# x = np.linspace(0, 5, 100)
# ax[0].plot(x, x ,'r-')
# ax[0].set_xlim(0, 5)
# ax[0].set_ylim(0, 5)
# ax[1].scatter(df['pps_prev'], df['pps'], s = 0.5)
# ax[1].plot(x, x,'r-')