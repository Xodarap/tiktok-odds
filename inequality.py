# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:55:05 2020

@author: bwsit
"""

import psycopg2
import itertools
import pandas
import numpy as np
import matplotlib.animation as animation

def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()
def lorenz_curve(X, annotation = None):
    X = np.array(X, dtype='int64')
    X.sort()
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    fig, ax = plt.subplots(figsize=[6,6])
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1) * 100, X_lorenz * 100)
    ## line plot of equality
    ax.plot(np.array([0,1]) * 100, [0,100], '--', color='k')
    ax.fill_between(np.arange(X_lorenz.size)/(X_lorenz.size-1) * 100, X_lorenz * 100)
    ax.set_ylabel('Cumulative % of followers')
    ax.set_xlabel('Cumulative % of population')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.text(40, 45, 'Line of equality', rotation = 45)
    ax.set_title('Follower Count Lorenz Curve')
    print(X_lorenz[int(X_lorenz.size * 0.8)])
    if annotation == 'equality':
        draw_lines(ax, 0.2, 0.2, 0.8, 0.8)
    if annotation == 'real':
        p1 = X_lorenz[int(X_lorenz.size * 0.9)]
        p2 = X_lorenz[int(X_lorenz.size * 0.99)]
        draw_lines(ax, 0.9, p1, 0.99, p2)

def draw_lines(ax, x1, y1, x2, y2):
     color = 'red'
     ax.axvline(x1 * 100, 0, y1, color = color)
     ax.axhline(y1 * 100, 0, x1, color = color)
     ax.text(-7, y1 * 100, f'{int(y1*100)}%', color = color)
     ax.text(x1 * 100 - 2, -7, f'{int(x1*100)}%', color = color)
     
     ax.axvline(x2 * 100, 0, y2, color = color)
     ax.axhline(y2 * 100, 0, x2, color = color)
     ax.text(-7, y2 * 100, f'{int(y2*100)}%', color = color)
     ax.text(x2 * 100 -2, -7, f'{int(x2*100)}%', color = color)

def inequality_per_year(year, column):
    cur.execute("""
    SELECT id, play_count, share_count, comment_count, like_count, create_time
    FROM public.tiktok_normalized
    where date_part('year', create_time) = (%s)
    and representative
                """, [year])
    result=cur.fetchall()
    data = [row[column] for row in result]
    lorenz_curve(data)
    coefficient = gini(data)
    
    return coefficient

def inequality_followers(year):
    cur.execute("""
        SELECT author, max(follower_count)
        from tiktok.users_normalized u
        inner join tiktok.videos_materialized v using (author)
        where date_part('year', v.create_time) = (2020)
        and v.representative
        group by author
                """, [year])
    result=cur.fetchall()
    data = [row[1] for row in result]
    lorenz_curve(data)
    coefficient = gini(data)
    
    return coefficient

#combinations = itertools.product([2014, 2015, 2016, 2017, 2018, 2019, 2020], [1, 2, 3, 4])
# =============================================================================
# result = [[year, inequality_per_year(year, 1), inequality_per_year(year, 2), 
#            inequality_per_year(year, 3), inequality_per_year(year, 4)] 
#           for year in [2014, 2015, 2016, 2017, 2018, 2019, 2020]]
# print(result)
# =============================================================================
result = [[2014, 0.7993713981573248, 0.9929475470165126, 0.8871729946065269, 0.6626561790560229], 
          [2015, 0.9458144214745937, 0.9926978985793387, 0.9426227195787098, 0.9433980123108631], 
          [2016, 0.975011410576089, 0.9801692770131308, 0.9626886028195223, 0.9705490307139615], 
          [2017, 0.9299705188465773, 0.9687253712727099, 0.9249392714026995, 0.9362435781785369], 
          [2018, 0.923189318555971, 0.9749903774646379, 0.9236212327969838, 0.9159047282126576], 
          [2019, 0.9710407699100051, 0.9841132580641035, 0.9743864190531284, 0.9720471946077368], 
          [2020, 0.9303255068781251, 0.962469297439619, 0.938871486688085, 0.9305082082343314]]
import matplotlib.pyplot as plt
result_frame = pandas.DataFrame(np.transpose(result), 
                                columns= [2014, 2015, 2016, 2017, 2018, 2019, 2020], 
                                index =['year','plays', "shares", "comments", 'likes'])
#inequality_per_year(2020, 1)
print(inequality_followers(2020))
# plt.plot('year', 'plays', data=result_frame)
# plt.show()

# =============================================================================
# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,
# fig1 = plt.figure()
# 
# # Fixing random state for reproducibility
# np.random.seed(19680801)
# 
# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)
# =============================================================================
    
    