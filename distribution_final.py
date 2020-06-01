# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:43:49 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
#from numpy import histogram
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import gamma
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy import optimize
from scipy.integrate import quad

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
    old_beta_est = alpha / beta #quad(lambda x: x*ben_gamma(x, alpha, beta), 0, 4)[0]
    new_beta_est = new_alpha / new_beta #quad(lambda x: x*ben_gamma(x, new_alpha, new_beta), 0, 4)[0]
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
            where author = 'charlidamelio'
            """)
personal_results = cur.fetchall()
alpha, beta = bootstrap(all_results)
new_alpha, new_beta = update(alpha, beta, personal_results)
display_table(alpha, beta, new_alpha, new_beta)

#update(prior, personal_results)