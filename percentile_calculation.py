# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:39:51 2020

@author: @lilweehag
"""
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy import optimize

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

def expected_views(n,views,aprior,bprior):
    n=int(n)
    aprior=float(aprior)
    bprior=float(bprior)    
    total=sum(np.log10(views))
    apos=aprior+n
    bpos=bprior+total
    b=apos/bpos
    lviews=1.42158/b
    if np.isneginf(lviews):
        return 0
    return lviews[0]

cur.execute("""
            SELECT distinct author
            FROM tiktok_normalized_table
            where representative = true
            and create_time >'2020-01-01'
            """)
all_results = cur.fetchall()
all_evs = []
count = 0
for row in all_results:
    author = row[0]
    cur.execute("""
            SELECT play_count
            FROM tiktok_normalized_table
            where author = ('%s')
            """ % (author))
    views = cur.fetchall()
    ev = expected_views(len(views), views,53.865, 83.099)
    all_evs.append(ev)
    count += 1
    if count > 1000:
        break
cur.close()
conn.close()

def get_percentiles(all_evs):
    plt.hist(all_evs, 100, facecolor='blue', alpha=0.5,density=True)
    all_evs.sort()
    entries_per_percentile = int(len(all_evs)/100)
    percentiles = [(percentile, all_evs[percentile * entries_per_percentile]) for percentile in range(1,100)]
    return percentiles

all_percentiles = []
for idx in range(0, 5):
    sampled = np.random.choice(all_evs, size = int(len(all_evs)/5))
    all_percentiles.append(get_percentiles(sampled))