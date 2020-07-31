# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:55:19 2020

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

cur.execute("""
            select m.*,
            case when sounds.title like '%original sound%' then 1 else 0 end original,
            case when users.followingcount < 10 then 1 else 0 end newb
            from m_sound_statistics m
            inner join sounds on sounds.id = soundid
            inner join users on users.nickname = sounds.authorname
where number_of_videos > 25
and first_video >= 1546300800
--order by vpl desc
""")

res=cur.fetchall()
colnames = [desc[0] for desc in cur.description]
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = colnames)

def examine_difference(column):
    orig = df[df[column] == 1]
    novel = df[df[column] == 0]
    plt.hist(np.log(orig['vpl']), density = True, bins = 200, alpha = 0.7, label = column)
    plt.hist(np.log(novel['vpl']), density = True, bins = 200, alpha = 0.7, label = f'not {column}')
    plt.legend()
    print(np.mean(np.log(orig['vpl'])))
    print(np.mean(np.log(novel['vpl'])))
    print(np.mean(novel['vpl']) / np.mean(orig['vpl']))
    print(stats.ttest_ind(np.log(orig['vpl']), np.log(novel['vpl'])))

examine_difference('original')
plt.figure()
examine_difference('newb')