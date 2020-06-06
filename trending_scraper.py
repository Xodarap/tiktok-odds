# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:07:52 2020

@author: 'bwsit
"""

from TikTokApi import TikTokApi
import psycopg2
import datetime
import json
conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

api = TikTokApi()

def saveTags(api, conn, cur):
    t_time=str(datetime.datetime.now())
    tags = api.discoverHashtags()
    count = 0

    for tag in tags:
        new_t=json.dumps(tag)
        cur.execute('INSERT INTO tiktok.trending_tags (fetch_time,json) VALUES (%s,%s);', (t_time,new_t))
        conn.commit()
        count += 1
    return count


def saveMusic(api, conn, cur):
    t_time=str(datetime.datetime.now())
    music = api.discoverMusic()
    count = 0

    for song in music:
        new_t=json.dumps(song)
        cur.execute('INSERT INTO tiktok.trending_music (fetch_time,json) VALUES (%s,%s);', (t_time,new_t))
        conn.commit()
        count += 1
    return count

tag_count = saveTags(api,conn,cur)
music_count = saveMusic(api,conn,cur)

print(f'Saved {tag_count} tags and {music_count} songs.')

cur.close()
conn.close()