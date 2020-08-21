
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TikTokApi import TikTokApi
import datetime
import time
import json
import os
import scipy.stats as stats
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from scipy.stats import kruskal

df = pd.read_csv("D:\\Downloads\\music data2.csv")
df['fv'] = df['fyp views']
df['vpl'] = df['final views'] / df['final like']

def plot_column(df, col):
    df.boxplot(column = col, by = 'music')
    mod = ols(f'{col} ~ music', data=df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(col)
    print(aov_table)
    
    rel = df.pivot(columns = 'music')
    stat, p = kruskal(*[rel[(col, mv)].dropna() for mv in ['yes', 'no', 'muted']])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    indep = ['view time percentage', #'final like', #'final comments','final shares', 
              'vpl']
    form = ' + '.join([f'Q("{i}")' for i in indep])
    mod = ols(f'{col} ~ {form}', data=df).fit()
    print(mod.summary())
    df['predicted'] = mod.predict(df[indep])
    df['error'] = df['predicted'] - df[col]
    rel = df.pivot(columns = 'music')
    stat, p = kruskal(*[rel[('error', mv)].dropna() for mv in ['yes', 'no', 'muted']])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    #df.boxplot(column = 'error', by = 'music')
    
    data = [df.loc[df['music'] == t, 'error'].values for t in ['no', 'muted', 'yes']]
    fig, ax = plt.subplots(1,1)
    ax.violinplot(data)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['None', 'Muted', 'With Volume'])
    ax.set_xlabel('Music')
    ax.set_ylabel('Error')
    rel = df.pivot(columns = 'music')
    stat, p = kruskal(*[rel[('error', mv)].dropna() for mv in ['yes', 'no', 'muted']])
    print('Error Statistics=%.3f, p=%.3f' % (stat, p))
    
    data = [df.loc[df['music'] == t, col].values for t in ['no', 'muted', 'yes']]
    fig, ax = plt.subplots(1,1)
    ax.violinplot(data)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['None', 'Muted', 'With Volume'])
    ax.set_xlabel('Music')
    ax.set_ylabel('FYP Views')

for threshold in [0, 5]:
    df_to_use = df[df['length'] > threshold]
    for col in ['fv']: #, 'views', 'vpl']:
        plot_column(df_to_use, col)