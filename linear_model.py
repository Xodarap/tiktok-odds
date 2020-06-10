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
from sklearn.preprocessing import PolynomialFeatures

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
cur.execute("""
            SELECT id, v.author, play_count, share_count, comment_count, like_count, 
            follower_count, following_count, bio ='', any_hashtags, any_tagged_users,
            is_commerce
            FROM tiktok.videos_materialized v
            inner join tiktok.users_normalized n on n.author = v.author
            where representative = true
            and create_time >= '2020-01-01'
            """)
all_results = cur.fetchall()
cur.close()
conn.close()
xlm=[]
ylm=[]

independent_variables = ['Shares', 'Comments', 'Likes', 'Follower Count',
                         'Following Count', 'Bio is empty', 'Any Hashtags',
                         'Any tagged users', 'Is Commerce'] 
used_variables = independent_variables

for r in all_results:
    if r[2]<1e5:
        xlm.append(list(r[3:3+len(independent_variables)]))
        ylm.append(r[2])
  
def process_results(xlm,ylm,single_transform = lambda x: x):
    X = xlm.applymap(single_transform)
    y = ylm.applymap(single_transform)
    independent_variables = list(X.columns)
    
    X_with_intercept = xlm.copy()
    X_with_intercept['Intercept'] = 1
    
    ols = sm.OLS(y.values, X_with_intercept.values)
    ols_result = ols.fit()
    
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]).fit(X, y)
    reg = LassoCV(cv=5, random_state=0, 
                  alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]).fit(X, y)
    
    plt.subplot(231)
    plt.plot(clf.predict(X), y, '.')
    #plt.plot(np.linspace(0,1e6), np.linspace(0,1e6), 'r')
    plt.title('Ridge')
    plt.ylabel('Actual')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Predicted')
    
    plt.subplot(232)
    plt.plot(reg.predict(X), y, '.')
    plt.title('Lasso')
    plt.yticks([])
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.xlabel('Predicted')
    
    plt.subplot(233)
    plt.plot(ols_result.fittedvalues, y, '.')
    plt.yticks([])
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.title('OLS')
    plt.xlabel('Predicted')
    
    simple_x = X_with_intercept[['Likes', 'Intercept']]
    simple_ols = sm.OLS(y.values, simple_x.values)
    simple_ols_result = simple_ols.fit()
    simple_output = [simple_ols_result.params[0] if var == 'Likes' else 'N/A' for var in independent_variables]
    
    plt.subplot(234)
    plt.plot(simple_ols_result.fittedvalues, y, '.')
    plt.yticks([])
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    plt.title('Simple OLS')
    plt.xlabel('Predicted')
    
    
    results = pd.DataFrame([[*reg.coef_, reg.intercept_, reg.alpha_, r2_score(reg.predict(X), y)],
                            [*clf.coef_[0], *clf.intercept_, clf.alpha_, r2_score(clf.predict(X), y)],
                            [*ols_result.params, 'N/A', r2_score(ols_result.fittedvalues, y)],
                            [*simple_output, simple_ols_result.params[1], 'N/A', r2_score(simple_ols_result.fittedvalues, y)]], 
                           index = ['Lasso', 'Ridge', 'OLS', 'OLS-Likes only'],
                           columns = independent_variables +
                                      ['Intercept', 'Alpha', 'R^2'])
    print(results)
    return results

def clean_df(df):
    return df.applymap(lambda v: 0 if (v is None or v == False) else 1 if v is True else v)

df = pd.DataFrame(xlm, columns = independent_variables)    
df_used = clean_df(df[used_variables])
ydf = pd.DataFrame(ylm)
plt.figure()
normal_results = process_results(df_used, ydf)
plt.figure()
log_results = process_results(df_used, ydf, lambda x: np.log(x) if x > 0 else 0)

poly = PolynomialFeatures(2, interaction_only = True)
poly_fit = poly.fit_transform(df_used, ydf)
poly_df = pd.DataFrame(poly_fit, columns = poly.get_feature_names(df_used.columns))
poly_df['Index'] = poly_df.index
df_used['Index'] = df_used.index
merged_df = pd.merge(poly_df, df_used, on = ['Index'] + used_variables)
plt.figure()
merged_results = process_results(merged_df, ydf)