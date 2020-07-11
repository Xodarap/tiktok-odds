# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.stats import lognorm
from scipy.stats import beta
from scipy.stats import uniform
from scipy.optimize import minimize
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from scipy.ndimage.filters import gaussian_filter1d

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
cur.execute("""
            SELECT play_count 
            FROM tiktok_normalized
            where representative = true
            and create_time >'2020-01-01'
            """)
all_results = np.array(cur.fetchall())
# cur.execute('''
#           select play_count 

# ''')
conn.close()

def create_ta_one(a1, b1, a2, b2, samples):
    return np.append(beta.rvs(a1, b1, size = samples), 
               beta.rvs(a2, b2, size = int(samples / 3)))
# def create_ta_two(a1, b1, a2, b2, samples):
#     target = beta.rvs(a1, b1, size = samples)
#     boost = b2 * binom.rvs(n = 1, p = a2, size = samples)
#     return target + boost * (1 - target)
def create_ta_two(a1, b1, a2, b2, samples = 50000, p = 1):
    target = beta.rvs(a1, b1, size = samples)
    actual = p * beta.rvs(a2, b2, size = samples)
    ta = actual / target
    #print(np.sum(ta >= 1))
    ta[ta >= 1] = beta.rvs(60, 3, size = np.sum(ta >= 1))
    return ta
def create_ta_two_pdf(a1, b1, a2, b2, samples = 50000, p = 1):
    def z_prob(z):        
        return ([beta.pdf(xv, a1, b1) * beta.pdf(xv/z, a2, b2)                       
                      for xv in np.linspace(0.1, 1, 100)])
    def p_z(z, include_gt):        
        return np.sum([beta.pdf(xv, a1, b1) * beta.pdf(xv/z, a2, b2)                       
                      for xv in np.linspace(0.1, 1, 100)]) + (
                              gt_prob * beta.pdf(z, 60, 3) if include_gt else 0)

    gt_prob = np.sum([p_z(z, False) for z in np.linspace(1, 15, 100)])
    ta_pdf = [p_z(z, True) for z in np.linspace(0, 1, 100)]
    #print(np.sum(ta >= 1))
    #ta[ta >= 1] = beta.rvs(60, 3, size = np.sum(ta >= 1))
    return ta_pdf
def create_histograms(a1, b1, a2, b2, v0, samples = 50000, ta_func = create_ta_one):
    ta = ta_func(a1, b1, a2, b2, samples)
    views = (1/(1-ta[ta != 1]))
    views = np.log10(views[views > 0])
    bins = 150
    values1, edges1 = np.histogram(v0*views, density = True, bins = bins)
    values2, edges2 = np.histogram(np.log10(all_results[all_results > 0]), 
                                   density = True, bins = edges1)
    return values1, edges1, values2, edges2

def loss(a1, b1, a2, b2, v0, samples = 50000, ta_func = create_ta_one):
    values1, _, values2, _ = create_histograms(a1, b1, a2, b2, v0, samples, ta_func)
    return np.sum(np.abs(np.power(values1 - values2, 1)))

def plot_histogram(ax, values1, edges1, values2, edges2):
    ysmoothed = gaussian_filter1d(values1, sigma=0.5)
    ax.plot(edges1[:len(edges1)-1], ysmoothed, label = 'predicted', color = 'orange')
    ax.bar(edges2[:len(edges2)-1], values2, label = 'actual', alpha = 0.7,
           width = edges2[1] - edges2[0])
    ax.set_xlabel('Log_10(Views)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 8)
    ax.legend()

def plot_density(ax, a1, b1, a2, b2, v0, samples = 50000, ta_func = create_ta_one):
    ta = ta_func(a1, b1, a2, b2, samples)
    plt.hist(ta, alpha = 0.7, bins = 100, density = True)
    ax.set_xlabel('Target view ratio / actual view ratio')
    ax.set_ylabel('Density')

fig, ax = plt.subplots(2, 1)
#res = create_histograms(5,2,60,4)
#plot_histogram(ax[0], * res)
calculate = False
ta_func = create_ta_two
#x0 = [5, 2, 60, 3, 4]
#x0 = [4.31116922, 2.22645846, 0.2, 0.9]
#x0 = [3, 2, 0.2, 0.8, 5]
#x0 = [8, 1.5, 2.8, 2, 4]
#x0 = [6, 4, 6.5, 9, 4]
#x0 = [4, 6, 3.5, 11, 6]
if calculate:
    optimal = minimize(lambda x: loss(*x, ta_func = ta_func), 
                       x0 = x0, method = 'Nelder-Mead',
                       bounds = [(0.1, 10), (0.1, 10),
                                 (0, 1), (0, 1),
                                 (1, 20)],
                       options = {'maxiter': 1})
    print(optimal)
    print(f'Loss: {loss(*optimal.x, ta_func = ta_func)}')
    params = optimal.x
else:
    params = np.array([ 3.13160471,  1.80981521, 11.07055068, 11.97540721,  4.00423733])
    #params = np.array([ 4.21872583,  2.21108072, 62.79339527,  3.00505674, 4])

plot_histogram(ax[0], *create_histograms(*params, samples = 50000, ta_func = ta_func))
plot_density(ax[1], *params, ta_func = ta_func)
x = np.linspace(0, 1, 100)
ax[1].plot(x, beta.pdf(x, params[0], params[1]), label = 'Target')
ax[1].plot(x, beta.pdf(x, params[2], params[3]), label = 'Actual')
plt.legend()

def expected_per_p(p, a1, b1, a2, b2):
    ta = create_ta_two(a1, b1, a2, b2, samples = 100000, p = p)
    views = (1/(1-ta[ta != 1]))
    return 2 * np.exp(np.mean(np.log(views))) / 6.01653 # normalize to 1
    #return 2 * np.sort(views)[int(len(views) / 2)] / 3.894
plt.figure()
expected = [expected_per_p(p, *params[:-1]) for p in x]
plt.plot(x, expected)
plt.xlabel('Relative Quality')
plt.ylabel('Relative Performance')

worse = len(list(filter(lambda v: v < 1, expected)))
plt.plot([0, worse/100], [1, 1], color = 'k')
plt.plot([worse/100, worse/100], [0, 1], color = 'k')
plt.annotate(f'Threshold: {int((1-(worse/100))*100)}% degredation', xy=(worse/100, 1),
             xytext = (0.2, 1.5), 
             arrowprops = dict(facecolor='black', shrink=0.05))
plt.xlim(0, 1.05)
plt.ylim(0, max(expected) + 0.1)
#plt.plot(x, create_ta_two_pdf(*params[:-1]))