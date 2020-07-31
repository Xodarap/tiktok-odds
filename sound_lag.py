

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
            from videos_delay_2
""")



df = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
conn.commit()
conn.close()
fig, ax = plt.subplots(1,1, sharex = True, figsize = (9, 14))
ax = [ax]
ax[0].scatter((df['rounded_delay']/10)/60/24, df['total_plays'], s = 5)
# ax[1].scatter(df['rounded_delay']/10, df['small_plays'], s = 0.5)
ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# ax[1].set_yscale('log')
ax[0].set_title('Views vs. Time After Sound Creation')
ax[0].set_xlabel('Days after sound creation')
ax[0].set_ylabel('Views')
ax[0].set_xlim(0, 300)
# ax[1].set_title('Views vs. Time After Sound Creation (Creators w/ <10k Followers)')
# ax[2].scatter(df['rounded_delay']/10, df['verified_plays'])
# ax[2].set_yscale('log')

# fig, ax = plt.subplots(3,1, sharex = True)
# ax[0].scatter(df['rounded_delay']/10, df['total_plays'] / df['total_likes'])
# ax[0].set_yscale('log')
# ax[1].scatter(df['rounded_delay']/10, df['small_plays'] / df['small_likes'])
# ax[1].set_yscale('log')

# fig, ax = plt.subplots(3,1, sharex = True)
# ax[0].scatter(df['rounded_delay']/10, df['total_likes'])
# ax[0].set_yscale('log')
# ax[1].scatter(df['rounded_delay']/10, df['small_likes'])
# ax[1].set_yscale('log')