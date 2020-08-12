from TikTokApi import TikTokApi
#import TikTokApi
import datetime
import time
import psycopg2
import json
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

def fetch():
    api = TikTokApi()
    tiktoks = api.byUsername('miss_anni21', count=50000)
    t_time=str(datetime.datetime.now())
    print(len(tiktoks))
    print(t_time)
    for t in tiktoks:
        new_t=json.dumps(t)
        cur.execute('INSERT INTO tiktok (time,json) VALUES (%s,%s)', (t_time,new_t))
        #outfile.write('%s\t%s\t%s\n' % (t['id'],t_time,t))
    conn.commit()
    cur.close()
    conn.close()

def make_subplots(nrow, ncol, **kw):
    fig, ax = plt.subplots(nrow, ncol, **kw)
    if nrow == 1 and ncol == 1:
        return fig, [ax]
    return fig, ax

cur.execute("select * from tiktok_normalized where author = 'addisonre'")
res=cur.fetchall()
df = pd.DataFrame(res, columns = [desc[0] for desc in cur.description])
df['vpl'] = df['play_count'] / df['like_count']
def make_plot(ax, df, col, title):
    ax.plot(df['create_time'], df[col])
    ax.set_yscale('log')
    article_date = pd.Timestamp('2019-12-02')
    vid_date = pd.Timestamp('2019-1-29')
    ax.plot([article_date, article_date], ax.get_ylim())
    ax.plot([vid_date, vid_date], ax.get_ylim())
    ax.set_title(title)

todo = [['play_count', 'Views'], 
 ['share_count','Shares'],
 ['comment_count', 'Comments'],
 ['like_count','Likes'],
 ['vpl', 'View/Like Ratio']]

# fig, axs = make_subplots(len(todo), 1, figsize=(6,10), sharex=True)
# for cinf, ax in zip(todo, axs):
#     make_plot(ax, df, *cinf)

# fig.tight_layout()
conn.close()
before1 = df[(df['create_time'] > pd.Timestamp('2020-07-03', tz='US/Pacific')) & (df['create_time'] < pd.Timestamp('2020-07-15', tz='US/Pacific'))]
before = df[(df['create_time'] > pd.Timestamp('2020-07-15', tz='US/Pacific')) & (df['create_time'] < pd.Timestamp('2020-07-27', tz='US/Pacific'))]
after = df[df['create_time'] >= pd.Timestamp('2020-07-27', tz='US/Pacific')]
stats.ks_2samp(before['play_count'], after['play_count'])
stats.ks_2samp(before1['play_count'], after['play_count'])
plt.violinplot([before1['play_count'].values, before['play_count'].values, after['play_count'].values])