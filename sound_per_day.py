# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:56:52 2020

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

cur.execute("""select m_sound_per_day.*, authorname
            from m_sound_per_day
            inner join sounds on sounds.id = soundid
            where soundid in (
            /*
                (select soundid
                from m_sound_per_day
                group by soundid
                order by sum(tm_vids) desc
                limit 20)  
                */
                	6796513206148909829 -- "Chinese New Year"
                ,6800996740322297858	--"Savage"
                ,6744446812653947654	--"Lottery"
            ) 
""")

result_df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.close()
def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax

def do_plot(ax, relevant, show_legend = False):
    l1 = ax.plot_date(relevant['d'], relevant['small_vids'].cumsum(), '-', label = '<10k Followers')
    ax2 = ax.twinx()
    l2 = ax2.plot_date(relevant['d'], relevant['tm_vids'].cumsum(), 'r-', label = '10M+ Followers')
    title = relevant['sound_title'].head(1).reset_index(drop = True)[0] + \
        ' - ' + \
        relevant['authorname'].head(1).reset_index(drop = True)[0]
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])
    ax2.set_yticks([])
    ax.set_ylabel('Cumulative # Of Videos')
    if show_legend:
        lns = l1+l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

def plot_per_vid(result_df):
    ids = result_df['soundid'].unique()
    # grouped = result_df.group_by(['soundid','d'])
    # fig, axs = make_subplots(5, 4, figsize = (15, 10))
    fig, axs = make_subplots(len(ids), 1, figsize = (6, 10))
    show_legend = True
    for idx, ax in zip(ids, axs.flatten()):
        relevant = result_df[result_df['soundid'] == idx]
        do_plot(ax, relevant, show_legend)
        show_legend = False
    axs[len(ids)-1].set_xlabel('Time')
    # axs[0].legend()
    fig.tight_layout()



plot_per_vid(result_df)
conn.close()
# fig, axs = make_subplots(1, 1)
# relevant = result_df.groupby('d')[['small_vids', 'tm_vids']].sum()
# relevant['d'] = relevant.index
# do_plot(axs[0], relevant)