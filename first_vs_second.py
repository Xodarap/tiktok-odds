# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:09:02 2020

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

xmin, xmax = plt.xlim()

conn=psycopg2.connect('dbname=postgres user=postgres password=0FFzm4282FW^')
cur = conn.cursor()

cur.execute("""SELECT play_count, author, row_number() over (partition by author order by create_time asc) rn, 
            like_count
from tiktok.videos_materialized
where representative""")
res=cur.fetchall()
result_df = pd.DataFrame(res, columns = ['Views', 'Author', 'Number', 'Likes'])
df = pd.DataFrame(columns = ['Views', 'First'])
df['Views'] = [np.log10(r[0]) if r[0] > 0 else 0 for r in res]
df['First'] = [r[2] for r in res]
df['Likes'] = [r[3] for r in res]

first_views = df[df['First'] == 1]['Views']
later_views = df[df['First'] == 15]['Views']
n_bins=200
plt.hist(first_views, n_bins, facecolor='blue', alpha=0.5,density=True, label = 'First videos')
plt.hist(later_views, n_bins, facecolor='red', alpha=0.5,density=True, label = 'Second videos')
plt.title('View counts of first and second videos')
plt.ylabel('Probability Density')
plt.xlabel('Log10(Views)')
plt.legend()
print(f'First average: {np.mean(first_views)}. Later average: {np.mean(later_views)}')
print(stats.ks_2samp(first_views, later_views))

plt.figure()

nonzero = df[df['Likes'] > 0]
nonzero['Ratio'] = nonzero['Views'] / nonzero['Likes']
first_ratio = nonzero[nonzero['First'] == 1]['Ratio']
second_ratio = nonzero[nonzero['First'] == 2]['Ratio']
n_bins=50
plt.hist(first_ratio, n_bins, facecolor='blue', alpha=0.5,density=True, label = 'First videos')
plt.hist(second_ratio, n_bins, facecolor='red', alpha=0.5,density=True, label = 'Second videos')
plt.title('View/like ratio of first and second videos')
plt.ylabel('Probability Density')
plt.xlabel('Log10(Views)/Likes')
plt.legend()
print(f'First average: {np.mean(first_ratio)}. Later average: {np.mean(second_ratio)}')
print(stats.ks_2samp(first_ratio, second_ratio))

def safe_log(x):
    if isinstance(x, Iterable):
        return [np.log10(v) if v > 0 else 0 for v in x]
    return np.log10(x) if x > 0 else 0

def log_gamma(x, alpha, beta):
    return sum([alpha * np.log(beta),
                       -gammaln(alpha), 
                      np.log(np.power(x, alpha - 1)), 
                      -beta * x])
def likelihood(data):
    alpha = 1.42158
    beta = 53.865/83.099
    return sum([log_gamma(d, alpha, beta) for d in data if d > 0])
def removeOutliers(x, outlierConstant):
    a = np.array(x)[np.isfinite(np.array(x))]
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

print(f'Likelihood first: {likelihood(first_views)} Likelihood second: {likelihood(later_views)}')

result_df['LogViews'] = safe_log(result_df['Views'])
first_views = result_df.loc[result_df['Number'] == 1,'Views']
second_views = result_df.loc[result_df['Number'] == 2,'Views']
print(f'First average: {np.mean(first_views)}. Later average: {np.mean(second_views)}')
print(stats.ks_2samp(first_views, second_views))

first = result_df[result_df['Number'] == 1]
second = result_df[result_df['Number'] == 2]
merged = first.merge(second, on = ['Author'])
merged['Ratio'] = merged['Views_y'] / merged['Views_x']
avg_ratio = np.mean(removeOutliers(merged['Ratio'],100))
plt.figure()
plt.hist(removeOutliers(merged['Ratio'], 3), n_bins, facecolor='blue', alpha=0.5,density=True)
plt.title('Second video / First video view count ratios')
plt.ylabel('Probability Density')
plt.xlabel('Second video views / First video views')
print(f'Ratio: {avg_ratio}.')
print(wilcoxon(merged['Views_x'], merged['Views_y']))

merged['Reasonable'] = merged['Views_x'] < 1e6
print(wilcoxon(merged.loc[merged['Reasonable'], 'Views_x'], merged.loc[merged['Reasonable'], 'Views_y']))

cur.close()
conn.close()