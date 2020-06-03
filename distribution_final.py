# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:43:49 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy import optimize

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

def log_gamma(x, alpha, beta):
    return np.exp(sum([alpha * np.log(beta),
                       -gammaln(alpha), 
                      np.log(np.power(x, alpha - 1)), 
                      -beta * x]))
def ben_gamma(x, alpha, beta = 1, location = 0, scale = 1):
    y = (x - location) / scale
    return log_gamma(y, alpha, beta) / scale
def update(alpha, beta, results):
    logviews = np.repeat([np.log10(r[1]) for r in results if r[1] > 0], 1)
    new_alpha = alpha + len(logviews)
    new_beta = beta + sum(logviews)
    x = np.linspace(0, 1, 100)
    plt.figure()
    prior_pdf = ben_gamma(x, alpha, beta)
    plt.plot(x, prior_pdf / sum(prior_pdf), label = 'prior')
    posterior_pdf = ben_gamma(x, new_alpha, new_beta)
    plt.plot(x, posterior_pdf / sum(posterior_pdf), label = 'posterior')
    plt.xlabel('beta')
    plt.ylabel('probability density')
    plt.legend()
    return (new_alpha, new_beta)

def bootstrap(results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    scales = []
    for i in range(0,1000):
        selected = [lv for lv in  np.random.choice(logviews, size = 200) if lv >= 1]
        density, bins = np.histogram(selected, 100, density=True)
        try:
            params, error = optimize.curve_fit(lambda x,b: ben_gamma(x, 1.4215833596, b, 1) , bins[1:100], density[1:100], 
                                               p0 = 1, maxfev = 10000)
        except:
            params = [0.67]
        scales.append(params[0])
    density, bins, p = plt.hist(scales, 100, facecolor='blue', alpha=0.5,density=True)
    params, standard = optimize.curve_fit(lambda x,a,b: ben_gamma(x, a, b, 0) , bins[1:100], density[1:100], 
                                          p0 = [1, 1], maxfev = 10000)
    
    x = np.linspace(0.5,1,100)
    plt.plot(x, ben_gamma(x, params[0], params[1], 0))
    plt.xlabel('beta')
    plt.ylabel('probability density')
    
    return params
        
def display_table(alpha, beta, new_alpha, new_beta):
    old_beta_est = alpha / beta 
    new_beta_est = new_alpha / new_beta 
    x = np.linspace(0,20,100)
    plt.figure()
    plt.plot(x, ben_gamma(x, 1.4215833596, old_beta_est), label = 'prior')
    plt.plot(x, ben_gamma(x, 1.4215833596, new_beta_est), label = 'posterior')
    plt.xlabel('log10(views)')
    plt.ylabel('probability density')
    plt.legend()
    old_expectation = 1.4215833596 / old_beta_est
    new_expectation = 1.4215833596 / new_beta_est
    print([[alpha, beta, old_beta_est, old_expectation, np.power(10, old_expectation)],
           [new_alpha, new_beta, new_beta_est, new_expectation, np.power(10, new_expectation)]])

def expected_views(n,total,aprior,bprior):
    n=int(n)
    total=float(total)
    aprior=float(aprior)
    bprior=float(bprior)
    total=n*np.log10(total/n)
    apos=aprior+n
    bpos=bprior+total
    b=apos/bpos
    lviews=1.42158/b
    views=10**lviews
    return views
    
'''
var n = parseInt($("#num").val());
var total = parseInt($('#total').val());
if(isNaN(total) || isNaN(n)) { return; }
total = n*Math.log10(total/n); // dodgy - assume each video got the same # of views
var alpha0 = 264.769 + n;
var beta0 = 405.25 + total;
var beta = alpha0/beta0;
var lviews = 1.42158 / beta;
var views = Math.pow(10, lviews);
$('#expectation').val(views);
'''


cur.execute("""
            SELECT id, play_count 
            FROM tiktok_normalized
            where representative = true
            and create_time >'2020-01-01'
            """)
all_results = cur.fetchall()

cur.execute("""
            SELECT id, play_count 
            FROM tiktok_normalized
            where author = 'benthamite'
            """)
personal_results = cur.fetchall()
alpha, beta = bootstrap(all_results)
new_alpha, new_beta = update(alpha, beta, personal_results)
display_table(alpha, beta, new_alpha, new_beta)
#v=expected_views(10,10000,alpha,beta)
#print(f'{v:,.2f}')
#print(f'{expected_views(10,1000,alpha,beta):,.2f}')
#print(f'{expected_views(10,100,alpha,beta):,.2f}')
#print(f'{expected_views(97,3435984275,alpha,beta):,.2f}')
