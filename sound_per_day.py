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

cur.execute("""select *
            from m_sound_per_day
            where soundid = 6743772128031557633
            --6787078702082607878
""")

result_df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.close()
def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax

fix, axs = make_subplots(2, 1, sharex = True)
axs[0].plot_date(result_df['d'], result_df['num_vids'])
axs[1].plot_date(result_df['d'], result_df['total_plays'])