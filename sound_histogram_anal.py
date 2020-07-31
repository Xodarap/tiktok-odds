# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:20:41 2020

@author: bwsit
"""

import matplotlib
# matplotlib.use("Agg")
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.special import gammaln
from scipy import stats
import pandas as pd
from collections.abc import Iterable
from scipy.stats import wilcoxon
import itertools

import matplotlib.animation as animation


from dotenv import load_dotenv
load_dotenv()
import os

conn=psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()

cur.execute("""select *
            from m_histogram_sound_anal_1
""")

result_df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.close()

def safe_convert(s):
    ret = int(s[0])
    if ret == 0:
        return 0
    return int(np.log10(ret))

# ranges = ['0-10','10-100', '100-1000', '1000-10000', '10000-100000',
#     '10000000-100000000',
#     '1000000-10000000',
#     '100000-1000000'
#     ]
# fake_data = pd.DataFrame(itertools.product(ranges, ranges, [0, 1]),
#                          columns = ['view_range', 'follower_range', 'original'])
# fake_data['count'] = np.random.randint(0, 100, size = (len(fake_data), 1))
# fake_data.drop(index = 1, inplace = True)
# result_df = fake_data
result_df['int_view_range'] = [safe_convert(s) for s in result_df['view_range'].str.split('-')]
result_df['int_follower_range'] = [safe_convert(s) for s in result_df['follower_range'].str.split('-')]
follower_ranges = pd.unique(result_df['int_follower_range'])
cols = result_df.columns
for fr in follower_ranges:
    for vr in pd.unique(result_df['int_view_range']):
        for orig in [0, 1]:
            if ~((result_df['int_follower_range'] == fr) & 
                        (result_df['int_view_range'] == vr) &
                        (result_df['original'] == orig)).any():
                result_df = result_df.append(pd.DataFrame([[orig, '','', 1, 1, vr, fr]],
                                              columns = cols))

def do_plot(ax, follower_range):
    fol_df = result_df[result_df['int_follower_range'] == follower_range]
    fol_df.sort_values(by = 'int_view_range', inplace = True)
    x = fol_df.loc[fol_df['original'] == 1, 'int_view_range']
    x = np.arange(8)
    width = 0.35
    fol_df.loc[fol_df['original'] == 1, 'count'] = fol_df.loc[fol_df['original'] == 1, 'count'] / np.sum(fol_df.loc[fol_df['original'] == 1, 'count'])
    fol_df.loc[fol_df['original'] == 0, 'count'] = fol_df.loc[fol_df['original'] == 0, 'count'] / np.sum(fol_df.loc[fol_df['original'] == 0, 'count'])
    l1 = ax.bar(x = x - width/2,
            height = fol_df.loc[fol_df['original'] == 1, 'count'],
            width = width,
            label = 'Original Sound', alpha = 1)
    l2 = ax.bar(x = x + width/2,
            height = fol_df.loc[fol_df['original'] == 0, 'count'],
            width = width,
            label = 'Existing Sound', alpha = 1)
    # plt.legend(loc = 'upper left')
    # ax.set_xlabel(f'Log(Views) - Follower Count: {follower_range}')
    ax.set_ylabel('Count of videos')
    # ax.set_yscale('log')
    disp_lower = np.power(10, follower_range)
    upper_range = 10 if follower_range == 0 else disp_lower * 10
    ax.set_title(f'Follower Range = {disp_lower}-{upper_range}')
    return (l1, l2)

def do_old():
    for bs in [0, 1]:
        fig, ax = plt.subplots(4, 1, sharex = True, figsize = (10, 10))     
        for idx, follower_range in enumerate(sorted(follower_ranges)[bs*4:(bs+1)*4]):
            do_plot(ax[idx], follower_range)
        ax[3].legend()
        ax[3].set_xlabel('Views')
        ax[3].set_xticklabels([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
        # fig.text(0.05, 0.5, 'Count of Videos', ha='center', va='center', rotation='vertical', fontsize=20)
        fig.tight_layout()

def animate(title, idx):
    global l1, l2, ax
    list_idx = idx % (len(follower_ranges) )
    follower_range = sorted(follower_ranges)[list_idx]
    fol_df = result_df[result_df['int_follower_range'] == follower_range]
    ax.set_ylim(0, max(fol_df['count']))
    # l1.set_ydata(fol_df.loc[fol_df['original'] == 1, 'count'])
    # l2.set_ydata(fol_df.loc[fol_df['original'] == 0, 'count'])
    for bar, height in zip(l1, fol_df.loc[fol_df['original'] == 1, 'count']):
        bar.set_height(height)
    for bar, height in zip(l2, fol_df.loc[fol_df['original'] == 0, 'count']):
        bar.set_height(height)
    # l1.set_height(fol_df.loc[fol_df['original'] == 1, 'count'])
    # l2.set_height(fol_df.loc[fol_df['original'] == 0, 'count'])
    disp_lower = np.power(10, follower_range)
    upper_range = 10 if disp_lower == 0 else disp_lower * 10
    print(f'Follower Range = {disp_lower}-{upper_range}')
    # ax.set_title(f'Follower Range = {follower_range}-{upper_range}')
    title.set_text(f'Follower Count = {disp_lower}-{upper_range}')
    return [*l1, *l2, title]
def do_animation():
    global l1, l2, ax
    fig, ax = plt.subplots()    
    plt.title('Histogram of Views by Follower Count and Sound')
    # ax.set_yscale('log')
    # ax.set_ylim(1, max(result_df['count']))
    title = ax.text(0.5,0.1, "",
                    transform=ax.transAxes, ha="center")
    l1, l2 = do_plot(ax, sorted(follower_ranges)[0])
    anim_func = lambda idx: animate(title, idx)
    ani = animation.FuncAnimation(
        fig, anim_func, interval=5000, blit=True, save_count=20)
    # plt.show()
    Writer = animation.writers['ffmpeg_file']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("D:/Downloads/movie.mp4", writer = writer)
    # str = ani.to_html5_video()
do_old()
# do_animation()