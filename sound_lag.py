

import psycopg2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LassoCV
from math import sqrt
from sklearn.metrics import r2_score
from scipy import stats
from dotenv import load_dotenv
load_dotenv()
import os

conn=psycopg2.connect(os.getenv("DB_URL"))
cur = conn.cursor()

cur.execute("""
            select * 
            from videos_delay_all_10k
""")



df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.commit()
conn.close()
fig, ax = plt.subplots(3,1, sharex = True, figsize = (9, 11))
ax = ax

df = df[df['rounded_delay'] >= 0] #todo
df['avg'] = df['total_plays'] / df['number_of_videos']
df['sem'] = 1.96 * df['total_plays_sd'].astype('float') / np.sqrt(df['number_of_videos'])
ax[0].plot(df['rounded_delay'], df['avg'])
ax[0].fill_between(df['rounded_delay'], df['avg'] - df['sem'], df['avg'] + df['sem'],
                   alpha = 0.3)
ax[0].set_title('Views vs. Time After Sound Creation')
ax[0].set_ylabel('Average Views')

df['avg_small'] = df['small_plays'] / df['small_count']
df['sem_small'] = 1.96 * df['small_plays_sd'].astype('float') / np.sqrt(df['small_count'])
ax[1].plot(df['rounded_delay'], df['avg_small'])
ax[1].fill_between(df['rounded_delay'], df['avg_small'] - df['sem_small'], df['avg_small'] + df['sem_small'],
                   alpha = 0.3)
ax[1].set_title('Views vs. Time After Sound Creation')
ax[1].set_ylabel('Average Views')

df['avg_big'] = df['big_plays'] / df['big_count']
df['sem_big'] = 1.96 * df['big_plays_sd'].astype('float') / np.sqrt(df['big_count'])
ax[2].plot(df['rounded_delay'], df['avg_big'])
ax[2].fill_between(df['rounded_delay'], df['avg_big'] - df['sem_big'], df['avg_big'] + df['sem_big'],
                   alpha = 0.3)
ax[2].set_title('Views vs. Time After Sound Creation')
ax[2].set_ylabel('Average Views')

conn.close()