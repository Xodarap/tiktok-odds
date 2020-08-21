from TikTokApi import TikTokApi
#import TikTokApi
import datetime
import time
import psycopg2
import json
from random import randint
from dotenv import load_dotenv
import os
load_dotenv()

conn2=psycopg2.connect(os.getenv("DB_URL"))
cur2 = conn2.cursor()
ids = [6861262732319051013,6861263376882863366,6861263613064154373,
        6861263701677198597,6861263945034976518, 6861265528732830982,
        6861304283304873221, 6861304523680402694]
api = TikTokApi(cookie_string = os.getenv("COOKIES"))

for i in ids:    
    t_time=str(datetime.datetime.now())
    ins = api.get_insights('benthamite', i)
    cur2.execute('insert into tiktok.insights values (%s, %s, %s)',
                (i, t_time,json.dumps(ins)))
    conn2.commit()
    
conn2.close()
#cat d:\Documents\tiktok_banque\TikTokApi.py | python
