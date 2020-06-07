# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:50:59 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
import statsmodels.api as sm

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
cur.execute("""
            SELECT *
            FROM tiktok_normalized_table
            where representative = true
            and create_time >= '2019-01-01'
            and create_time < '2020-01-01'
            """)
all_results = cur.fetchall()
cur.close()
xlm=[]
ylm=[]
    
for r in all_results:
    if r[2]<1e6:
        xlm.append([r[3],r[4],r[5]])
        ylm.append(r[2])


def transform(value, single_transform = lambda x: x):
    if isinstance(value, list):
        result = []
        for v in value:
            if v > 0:
                result.append(single_transform(v))
            else:
                result.append(0)
        return result
    else:
        if value == 0:
            return 0
        return single_transform(value)
    
def process_results(xlm,ylm,single_transform = lambda x: x):
    xlm = [transform(row,single_transform) for row in xlm]
    ylm = [transform(row,single_transform) for row in ylm]
    X=pd.DataFrame(xlm,columns=['shares','comments','likes'])
    y=pd.DataFrame(ylm,columns=['views'])
    
    N = len(X)
    p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, p-1] = 1
    X_with_intercept[:, 0:(p-1)] = X.values
    
    ols = sm.OLS(y.values, X_with_intercept)
    ols_result = ols.fit()
    
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]).fit(X, y)
    reg = LassoCV(cv=5, random_state=0).fit(X, y)
    
    plt.subplot(131)
    plt.plot(clf.predict(X), y, '.')
    #plt.plot(np.linspace(0,1e6), np.linspace(0,1e6), 'r')
    plt.title('Ridge')
    plt.ylabel('Actual')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Predicted')
    
    plt.subplot(132)
    plt.plot(reg.predict(X), y, '.')
    plt.title('Lasso')
    plt.yticks([])
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Predicted')
    
    plt.subplot(133)
    plt.plot(ols_result.fittedvalues, y, '.')
    plt.yticks([])
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.title('OLS')
    plt.xlabel('Predicted')
    
    
    results = pd.DataFrame([[*reg.coef_, reg.intercept_, reg.alpha_, r2_score(reg.predict(X), y)],
                            [*clf.coef_[0], *clf.intercept_, clf.alpha_, r2_score(clf.predict(X), y)],
                            [*ols_result.params, 'N/A', ols_result.rsquared]], 
                           index = ['Lasso', 'Ridge', 'OLS'],
                           columns = ['Shares', 'Comments', 'Likes', 'Intercept', 'Alpha', 'R^2'])
    print(results)

plt.figure()
process_results(xlm, ylm)
plt.figure()
process_results(xlm, ylm, np.log)