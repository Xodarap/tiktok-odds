# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:07:52 2020

@author: 'bwsit
"""

from TikTokApi import TikTokApi
from TikTokApi import browser
import psycopg2
import datetime
import json
import time
from random import randint
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

api = TikTokApi()

class BenTikTokApi(TikTokApi):
    def getUser(self, username, language='en'):
        api_url = "https://m.tiktok.com/api/user/detail/?uniqueId={}&language={}&verifyFp=".format(
            username, language)
        b = browser.browser(api_url)
        return self.getData(api_url, b.signature, b.userAgent)

ben_api = BenTikTokApi()

cur.execute('''
select distinct author from tiktok_normalized_table
where author not in (
    select distinct json->'userInfo'->'user'->>'uniqueId'
    from tiktok.users
	where json->'userInfo'->'user'->>'uniqueId' is not null
)
and create_time >= '2020-01-01'
''')
all_results = cur.fetchall()
for row in all_results:
    author = row[0]
    print(f'Getting {author}')
    t_time=str(datetime.datetime.now())
    try:
        userData = ben_api.getUser(author)
        new_t=json.dumps(userData)
        print(new_t)
        cur.execute('INSERT INTO tiktok.users (fetch_time,json) VALUES (%s,%s);', (t_time,new_t))
        conn.commit()
    except:
        print('Error')
        continue
    time.sleep(randint(1, 10))
#print(f'Saved {tag_count} tags and {music_count} songs.')

cur.close()
conn.close()