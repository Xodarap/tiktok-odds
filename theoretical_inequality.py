# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 07:28:12 2020

@author: bwsit
"""



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import pareto

def multiply_plot(pdf1, title1, pdf2, title2):
    x = np.linspace(0, 10, 100)
    fig = plt.figure(figsize = (5, 7))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(x, pdf1(x))
    ax1.set_title(title1)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(x, pdf2(x))
    ax2.set_title(title2)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan = 2)
    prod_pdf = lambda z: pdf1(z) * pdf2(z)
    ax3.plot(x, prod_pdf(x))
    prod_name = title1 + ' x ' + title2
    ax3.set_title('Success = ' + prod_name)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
    return  prod_pdf, prod_name

norm_pdf = lambda x: norm.pdf(x, 5, 2)
prod_pdf, prod_name = multiply_plot(norm_pdf, 'Talent', norm_pdf, 'Effort')
prod_pdf, prod_name = multiply_plot(prod_pdf, prod_name, norm_pdf, 'Beauty')
pareto_pdf = lambda x: pareto.pdf(x, 5, 2)
prod_pdf, prod_name = multiply_plot(prod_pdf, prod_name, pareto_pdf, 'IG Followers')
