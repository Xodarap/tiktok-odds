# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:22:04 2020

@author: bwsit
"""

import psycopg2
import numpy as np
import pandas as pd
from clarifai.rest import ClarifaiApp
import json

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""        
            select distinct recommended_username username, recommended_pfp
            from tiktok.race_check
            where recommended_username not in (select username from tiktok.clarifai_data)
            and recommended_pfp is not null
            union all
            select distinct source_username username, source_pfp
            from tiktok.race_check
            where source_username not in (select username from tiktok.clarifai_data)
            and source_pfp is not null
""")

res=cur.fetchall()

class ConceptDictionary:
    def __init__(self, concepts):
        self.concepts = { c['name'] : c for c in concepts }

    def concept_valid(self, name, threshold = 0.75):
        return self.concepts[name]['value'] >= threshold
    
    def gender(self):
        if self.concept_valid('masculine'):
            return 'Male'
        if self.concept_valid('feminine'):
            return 'Female'
        return 'Unknown'
    
    def race(self):
        races = ['hispanic, latino, or spanish origin', 'middle eastern or north african',
                 'american indian or alaska native', 'black or african american', 
                 'native hawaiian or pacific islander', 'white', 'asian']
        for race in races:
            if self.concept_valid(race):
                return race
        return 'Unknown'
    

app = ClarifaiApp(api_key='')
model = app.models.get('demographics')

def get_user(username, pfp):
    response = model.predict_by_url(url=pfp)
    try:
        concepts = response['outputs'][0]['data']['regions'][0]['data']['concepts']
        dictionary = ConceptDictionary(concepts)
        cur.execute('INSERT INTO tiktok.clarifai_data (username,pfp_url,response,gender,race) VALUES (%s,%s,%s,%s,%s)', 
                (username, pfp, json.dumps(response), dictionary.gender(), 
                 dictionary.race()))
    except:
        cur.execute('INSERT INTO tiktok.clarifai_data (username,pfp_url,response,gender,race) VALUES (%s,%s,%s,%s,%s)', 
                (username, pfp, json.dumps(response), 'Unknown', 'Unknown'))
    conn.commit()

for row in res:
    username = row[0]
    pfp = row[1]
    print(username)
    get_user(username,pfp)
conn.close()