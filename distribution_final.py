# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:43:49 2020

@author: bwsit
"""


import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import gamma
from scipy.special import gamma as gamma_function
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy import optimize
from scipy.integrate import quad

conn=psycopg2.connect('dbname=postgres user=postgres password=0FFzm4282FW^')
cur = conn.cursor()

class TDistribution:
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
    
    def update(self, new1, new2):
        for i in range(0,4):
            self.d1[i] = 1

def fit_data_compound_beta(results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    plt.figure()
    #histogram
    n_bins=100
    density, bins, p = plt.hist(logviews, n_bins, facecolor='blue', alpha=0.5,density=True)
    
    #fit distributions
    cutoff = 3
    a, b, loc, shape = beta.fit([lv for lv in logviews if lv < cutoff])
    a2, b2, loc2, shape2 = beta.fit([lv for lv in logviews if lv >= cutoff-0.2])
    
    #plot distributions
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = beta.pdf(x, a, b, loc, shape)
    #plt.plot(x, p / 2, '--b', linewidth=1)
    p2 = beta.pdf(x, a2, b2, loc2, shape2)
    #plt.plot(x, p2 / 2, '--r', linewidth=1)
    plt.plot(x, (p + p2) / 2, 'k', linewidth=1)
    plt.xlabel('log10(views)')
    plt.ylabel('probability density')

def fit_data_exponential(results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    loc, scale = expon.fit([lv for lv in logviews if lv >= 1])
    n_bins=100
    density, bins, p = plt.hist(logviews, n_bins, facecolor='blue', alpha=0.5,density=True)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = expon.pdf(x, loc, scale)
    plt.plot(x, p, 'r')
    

def fit_data_gamma(results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    a, loc, scale = gamma.fit([lv for lv in logviews if lv >= 1])
    n_bins=100
    density, bins, p = plt.hist(logviews, n_bins, facecolor='blue', alpha=0.5,density=True)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)    
    #p = gamma.pdf(x, a, loc, scale)
    p = ben_gamma(x, a, 1, loc, scale)
    plt.plot(x, p, 'r')
    #x = np.linspace(0.1, 10, 100)
    #plt.scatter(gamma.pdf(x, a, loc, scale), ben_gamma(x, a, 1, loc, scale))

def ben_fit_exponential(data):
    def fit(l):
        likelihoods = [np.exp(-l * d) for d in data]
        total = sum(np.log(likelihoods))
        return total
    print(fit(0.5))
    print(fit(0.4))
    print(fit(0.3))
    print(fit(0.6))
    best_lambda = minimize(fit, 0.5)
    return best_lambda

def ben_normal_gamma(x, alpha, beta = 1):
    return (np.power(beta, alpha)/gamma_function(alpha)) * np.power(x, alpha - 1) * np.exp(-beta * x)
def log_gamma(x, alpha, beta):
    return np.exp(sum([alpha * np.log(beta),
                       -gammaln(alpha), 
                      np.log(np.power(x, alpha - 1)), 
                      -beta * x]))
def ben_gamma(x, alpha, beta = 1, location = 0, scale = 1):
    y = (x - location) / scale
    return log_gamma(y, alpha, beta) / scale
def posterior(l, alpha, beta, n, xbar):
    return np.power(l, alpha + n - 1) * np.exp(-l * (beta + n + xbar))
def gamma_2(x, k, theta):
    return np.power(x, k-1) * np.exp(-x/theta) / (gamma_function(k) * np.power(theta, k))
def update(alpha, beta, results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    new_alpha = alpha + len(logviews)
    new_beta = beta + sum(logviews)
    x = np.linspace(0, 10, 100)
    plt.clf()
    plt.plot(x, ben_gamma(x, alpha, beta, 1, 1), label = 'prior')
    plt.plot(x, ben_gamma(x, new_alpha, new_beta, 1, 1), label = 'posterior')
    plt.xlabel('lambda')
    plt.ylabel('probability density')
    plt.legend()
    return (new_alpha, new_beta)

def bootstrap(results):
    logviews=[np.log10(r[1]) for r in results if r[1] > 0]
    scales = []
    for i in range(0,1000):
        selected = np.random.choice(logviews, size = 1000)
        loc, scale = expon.fit([lv for lv in selected if lv >= 1], floc = 1)
        scales.append(scale)
    density, bins, p = plt.hist(scales, 100, facecolor='blue', alpha=0.5,density=True)
    params, standard = optimize.curve_fit(lambda x,a,b: ben_gamma(x, a, b, 1) , bins[1:100], density[1:100], 
                                     p0 = [1, 1], maxfev = 5000)
    plt.plot(np.linspace(0,10,100), ben_gamma(np.linspace(0,10,100), params[0], params[1], 1))
    plt.xlabel('lambda')
    plt.ylabel('probability density')
    
    return params
        
def display_table(alpha, beta, new_alpha, new_beta):
    lambda_old = quad(lambda x: x*ben_gamma(x, alpha, beta, 1, 1), 1, 4)[0]
    lambda_new = quad(lambda x: x*ben_gamma(x, new_alpha, new_beta, 1, 1), 1, 4)[0]
    x = np.linspace(0,10,100)
    #old_exp = lambda k: lambda_old * np.exp(-lambda_old * (k - 1))
    #new_exp = lambda k: lambda_new * np.exp(-lambda_new * (k - 1))
    plt.figure()
    plt.plot(x, expon.pdf(x, 1, lambda_old), label = 'prior')
    plt.plot(x, expon.pdf(x, 1, lambda_new), label = 'posterior')
    plt.legend()
    print([[alpha, beta, alpha/beta, lambda_old, np.power(10, expon.mean(1, lambda_old))],
           [new_alpha, new_beta, new_alpha/new_beta, lambda_new, np.power(10, expon.mean(1, lambda_new))]])

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
#update(prior, personal_results)