# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:12:59 2020

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
from matplotlib.ticker import ScalarFormatter
#from fbprophet import Prophet


from dotenv import load_dotenv
load_dotenv()
import os

conn=psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()

cur.execute("""
            select * 
            from m_duration
            where duration <= 60
            and duration != 0
""")



df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.commit()
conn.close()
fig, ax = plt.subplots(2,1, sharex = True, figsize = (9, 14))
# ax = [ax]
def do_plot(ax, column, title, ylab):
    ax.plot(df['duration'], df[column], label = ylab)
    z = np.polyfit(df['duration'].astype('int32'), df[column].astype('int32'), 1)
    p = np.poly1d(z)
    ax.plot(df['duration'], p(df['duration']), 'r--', label = 'Trendline')
    # ax.set_yscale('log')
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.legend()

do_plot(ax[0], 'plays_avg', 'Views vs. Duration', 'Views')
do_plot(ax[1], 'vpl_avg', 'Views Per Like vs. Duration', 'VPL')
ax[1].set_xlabel('Duration (seconds)')
ax[0].ticklabel_format(axis = 'y', style = 'sci')
# for axi, column in zip(ax, ['plays_avg', 'number_of_videos', 'likes_avg', 'vpl_avg']):
#     do_plot(axi, column)
# ax[0].plot(df['duration'], df['plays_avg'])
# # ax[0].plot(df['duration'], df['plays_avg'] - np.sqrt(df['plays_var']), 'r--')
# ax[1].plot(df['duration'], df['number_of_videos'])
# ax[2].plot(df['duration'], df['likes_avg'])
# ax[3].plot(df['duration'], df['vpl_avg'])
# # ax[0].plot(df['duration'], df['plays_avg'] + np.sqrt(df['plays_var']), 'r--')
# # ax[1].scatter(df['rounded_delay']/10, df['small_plays'], s = 0.5)
# for idx in range(0, 4):
#     ax[idx].set_yscale('log')
# # ax[0].set_xscale('log')
# # ax[1].set_yscale('log')
# ax[0].set_title('Views vs. Time After Sound Creation')
# ax[0].set_xlabel('Days after sound creation')
# ax[0].set_ylabel('Views')