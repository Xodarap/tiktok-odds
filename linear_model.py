# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:50:59 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy import optimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
#from sklearn.model_selection import GridScearchCV

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
cur.execute("""
            SELECT *
            FROM tiktok_normalized_table
            where representative = true
            and create_time >= '2020-01-01'
            """)
print(cur.description)
all_results = cur.fetchall()
df=[]
xlm=[]
ylm=[]
def transform(value):
    single_transform = np.log
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
    
for r in all_results:
    if r[2]<1e7:
        xlm.append(transform([r[3],r[4],r[5]]))
        ylm.append(transform(r[2]))
        df.append([(r[2]),r[3],r[4],r[5]])
df=pd.DataFrame(df,columns=['views','shares','comments','likes'])
features = ['views','comments','shares','likes']

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['views']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pca1', 'pca2'])
finalDf = pd.concat([principalDf, df[['views']]], axis = 1)

"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Views', fontsize = 15)
ax.set_title('PCA1 vs Views', fontsize = 20)
ax.scatter(finalDf['pca1'],finalDf['views'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Views', fontsize = 15)
ax.set_title('PCA2 vs Views', fontsize = 20)
ax.scatter(finalDf['pca2'],finalDf['views'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA1', fontsize = 15)
ax.set_ylabel('PCA2', fontsize = 15)
ax.set_title('PCA vs Views', fontsize = 20)
ax.scatter(finalDf['pca1'],finalDf['pca2'],c=finalDf['views'])
"""
'''
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Comments', fontsize = 15)
ax.set_ylabel('Views', fontsize = 15)
ax.set_title('Comments vs Views', fontsize = 20)
ax.scatter(df['comments'],df['views'])
'''
'''
lm = linear_model.LinearRegression()
indep_vars = df[['comments', 'shares', 'likes']]
model = lm.fit(indep_vars, df['views'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Predicted views', fontsize = 15)
ax.set_ylabel('Actual Views', fontsize = 15)
ax.set_title('Predicted vs Actual Views', fontsize = 20)
ax.scatter(lm.predict(df[['comments', 'shares', 'likes']]),df['views'])
print(f"R^2: {lm.score(indep_vars, df['views'])}")
#x_range = np.linspace(0,1400000,100)
#ax.plot(x, lm.predict(x))
'''
X=pd.DataFrame(xlm,columns=['shares','comments','likes'])
y=pd.DataFrame(ylm,columns=['views'])
model = linear_model.LinearRegression()
model.fit(X=X, y=y)

N = len(X)
p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term

X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, p-1] = 1
X_with_intercept[:, 0:(p-1)] = X.values

import statsmodels.api as sm
ols = sm.OLS(y.values, X_with_intercept)
ols_result = ols.fit()
print(ols_result.summary())
cur.close()

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['views'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['likes'],y_pred, '.')
        #plt.plot(data['likes'],data['views'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['views'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

#Initialize predictors to be set of 15 powers of x
#predictors=['x']
#predictors.extend(['x_%d'%i for i in range(2,16)])

predictors = ['shares','comments','likes']

#Set the different values of alpha to be tested
alpha_ridges = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['shares','comments','likes']
ind = ['alpha_%.2g'%alpha for alpha in alpha_ridges]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
idx = 0
#for alpha in alpha_ridges:
#    coef_matrix_ridge.iloc[idx,] = ridge_regression(df, predictors, alpha, models_to_plot)
#    idx += 1

clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 5, 10, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]).fit(X, y)
reg = LassoCV(cv=5, random_state=0).fit(X, y)

plt.subplot(131)
plt.plot(clf.predict(X), y, '.')
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

#ridge_regressor = GridSearchCV(ridge, scoring = 'neg_mean_squared_error', cv = 5)
#ridge_regressor.fig(X, y)