# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:59:41 2020

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
#from fbprophet import Prophet

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""
            --REFRESH MATERIALIZED VIEW tiktok.videos_all_materialized;
            select d1.pps current_pps, d2.*
            /*d1.id, d1.pps, d1.ppl, d1.ppe, play_count_2, elapsed_seconds_2,
            like_count_2, d_seconds, fetch_time_1, fetch_time_2, lps*/
            from tiktok.videos_delta d1
            inner join tiktok.videos_delta d2 on d1.id = d2.id and
            d1.index_2 = d2.index_1
            where d1.id in 
            (select distinct id
            from tiktok.videos_delta
            where d_play > 0
            group by id
            having count(1) > 20)
            and d1.id in (6838386972445248773, 6844191231224876293)
            and d1.d_seconds < 800
            and d1.elapsed_seconds_2 < 30000
            order by d1.id, d1.elapsed_seconds_2 asc
""")

res=cur.fetchall()
colnames = [desc[0] for desc in cur.description]
conn.commit()
conn.close()
df = pd.DataFrame(res, columns = colnames)
#autocorrelation_plot(df['pps'])
#plot_acf(df['pps'], lags=10)
def AR():
    model = AutoReg(train, lags=5)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=6, end=len(train)-1, dynamic=False)
    #for i in range(len(predictions)):
    #	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    rmse = sqrt(mean_squared_error(test[6:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot results
    plt.plot(test)
    plt.plot(predictions, color='red')

def ARIMA():
    series = train
    X = df
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = train['pps']
    predictions = list()
    for t in range(len(test)):
    	model = ARIMA(history, exog = train['lps'], order=(5,1,0), dates = df['fetch_time_2'])
    	model_fit = model.fit(disp=0)
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    	print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
def prophet():
    df['Y'] = df['pps']
    df['DS'] = df['fetch_time_2']
    m = Prophet()
    m.fit(df)


#df.fillna(inplace = True)
def simple(df, ax, reg = None):
    y = df['current_pps']    
    X = df[independent_variables]
    X.fillna(0, inplace = True)
    if reg is None:
        reg = LassoCV(cv=5, random_state=0, 
              alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]).fit(X, y)
    
        results = pd.DataFrame([[*reg.coef_, reg.intercept_, reg.alpha_, r2_score(reg.predict(X), y)]],
                               columns = independent_variables +
                                              ['Intercept', 'Alpha', 'R^2'])
        print(results)
    def total_views(pps, df):
        return pps * [d.total_seconds() for d in df['d_time']]
    df['prev_d_plays'] = df['d_play'].shift(1)    
    df['extrap_plays'] = np.cumsum(total_views(y, df))
    df['prev_extrap_plays'] = df['extrap_plays'].shift(1)  
    df['naive_estimate'] = df['prev_extrap_plays'] + df['prev_d_plays']
    ax.plot(df['elapsed_seconds_2'] / 60, df['extrap_plays'], label = 'actual')
    ax.plot(df['elapsed_seconds_2'] / 60, np.cumsum(total_views(reg.predict(X), df)), label = 'predicted')
    # ax.plot(df['elapsed_seconds_2'] / 60, df['naive_estimate'], label = 'naive estimate')
    ax.set_ylabel('Views')
    ax.legend()
    return reg

df1 = df[df['id'] == 6838386972445248773]   
df2 = df[df['id'] == 6844191231224876293]   
#independent_variables = ['lps', 'pps', 'ppl', 'cps', 'ppe']
independent_variables = ['lps']
fig, ax = plt.subplots(2,1)
reg = simple(df1, ax[0])
simple(df2, ax[1], reg)
ax[1].set_xlabel('Minutes since posting')

