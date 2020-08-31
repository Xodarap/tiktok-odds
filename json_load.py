from TikTokApi import TikTokApi
#import TikTokApi
from datetime import datetime
import time
import psycopg2
import json
from random import randint
from dotenv import load_dotenv
import os
load_dotenv()
conn2=psycopg2.connect(os.getenv("DB_URL"))
cur2 = conn2.cursor()
for filename in os.listdir('D:\\Documents\\tiktok_banque\\v4'):
    if filename.endswith(".json") : 
        f = open('D:\\Documents\\tiktok_banque\\v4\\' + filename, 'rb')
        data = json.load(f)
        f.close()
        try:
            t_time = datetime.fromtimestamp(int(data['extra']['now'])/1000)
            vid_id = data['video_info']['aweme_id']
            cur2.execute('insert into tiktok.insights values (%s, %s, %s)',
            (vid_id, t_time,json.dumps(data)))
        except Exception as ex:
            print(ex)
            print(f'Err: {filename}')
conn2.commit()
conn2.close()