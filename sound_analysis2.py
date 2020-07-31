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
            case when title like '%original sound%' then 1 else 0 end original
            from m_sound_statistics_verified m
where number_of_videos > 100
and first_video_ts >= 1546300800

--order by vpl desc
""")

res=cur.fetchall()
colnames = [desc[0] for desc in cur.description]
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = colnames)
df['NonV-VPL'] = (df['plays'] - df['verified_plays']) / (df['likes'] - df['verified_likes'])
df['NonF-VPL'] = (df['plays'] - 0.05 * df['total_followers']) / (df['likes'])
#df.loc[df['sound_creator_verified'] == None, 'sound_creator_verified'] = 01679363
df['sound_creator_verified'].fillna(value = False, inplace=True)

def safe_log(x):
    out = np.log(x)
    return out[~np.isnan(out)]

def make_plot(column, orig, novel, metric):
    plt.hist(safe_log(orig[metric]), density = True, bins = 200, alpha = 0.7, label = column)
    plt.hist(safe_log(novel[metric]), density = True, bins = 200, alpha = 0.7, label = f'not {column}')
    # plt.xscale('log')
    plt.legend()
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.figure()    

def examine_difference(column, metric):
    safe = df[np.logical_and(df[metric] > 0, df[metric] < 100)]
    orig = safe[safe[column] == 1]
    novel = safe[safe[column] == 0]
    logged_orig = safe_log(orig[metric])
    logged_novel = safe_log(novel[metric])
    # make_plot(column, logged_orig, logged_novel, metric)
    make_plot(column, orig, novel, metric)
    return [np.mean(np.log(orig[metric])), np.mean(np.log(novel[metric])),
            np.mean(orig[metric]) / np.mean(novel[metric]) , 
            stats.ttest_ind(logged_orig, logged_novel).pvalue,
            stats.ttest_ind((orig[metric]), (novel[metric])).pvalue]


'''('original', 'NonF-VPL'),
        ('sound_creator_verified', 'NonV-VPL'),
        ('sound_creator_verified', 'vpl'),
        ('original', 'vpl'),'''
pairs = [    
        ('original', 'small_vpl'),
        ('original', 'ok_vpl'),
        ('original', 'fk_vpl'),
        ('original', 'ohk_vpl'),
        ('original', 'vpl')]
pairs = [    
        ('original', 'small_plays'),
        ('original', 'ok_plays'),
        ('original', 'fk_plays'),
        ('original', 'ohk_plays'),
        ('original', 'plays')]
results = [examine_difference(*p) for p in pairs]
result_df = pd.DataFrame(results, columns = ['Mean Log 1', 'Mean log 0',
                                             '1/0', 'p log', 'p'])
# plt.scatter(np.log(df['number_of_videos']), np.log(df['vpl']))

#examine_difference('newb', 'NonV-VPL')