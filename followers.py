# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:35:52 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LassoCV
from math import sqrt
from sklearn.metrics import r2_score
from scipy import stats
#from fbprophet import Prophet


from dotenv import load_dotenv
load_dotenv()
import os

conn=psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()
cur.execute('select * from insight_users where unique_id = \'benthamite\'')
res=cur.fetchall()
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax
fig, axs = make_subplots(2, 1, figsize = (6, 10))
axs[0].plot_date(df['fetch_date'], df['follower_count'], 'r-', label = 'Actual')
start = df['fetch_date'].min()
rng = pd.date_range(start, '2020-12-31')
required_growth = 100000- df.loc[0, 'follower_count']
required_daily = required_growth / len(rng)
required_cum = np.repeat(required_daily, len(rng)).cumsum() + df.loc[0, 'follower_count']
axs[0].plot_date(rng, required_cum, '--', label = 'Required')
axs[0].set_title('Cumulative Follower Growth')

df['shifted_followers'] = df['follower_count'].shift(1)
df['shifted_date'] = df['fetch_date'].shift(1)
df['date_delta'] = (df['shifted_date'] - df['fetch_date']) / np.timedelta64(1, 'D')
df['fpd'] = (df['shifted_followers'] - df['follower_count']) / df['date_delta']
axs[1].plot_date(df['fetch_date'], df['fpd'], 'r-', label = 'Actual')
axs[1].plot_date(df['fetch_date'], np.repeat(required_daily, len(df['fetch_date'])),
                 '--', label = 'Actual')
axs[1].set_title('Daily Follower Growth')
axs[1].set_yscale('log')
fig.tight_layout()