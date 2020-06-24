# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:25:23 2020

@author: bwsit
"""


import psycopg2
import numpy as np
import pandas as pd
from clarifai.rest import ClarifaiApp
import json
from scipy.stats import chi2_contingency

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""        
select source_username, source.gender, source.race,
    recommended_username, rec.gender, rec.race
    from tiktok.race_check
    inner join tiktok.clarifai_data source on source.username = source_username
    inner join tiktok.clarifai_data rec on rec.username = recommended_username
""")

res=cur.fetchall()
df = pd.DataFrame(res, columns = ['source_username', 'source.gender', 'source.race',
    'recommended_username', 'rec.gender', 'rec.race'])

def run_analysis(df):
    gender = df[np.all([df['source.gender'] != 'Unknown', df['rec.gender'] != 'Unknown'], 0)]
    gender_table = pd.crosstab(gender['source.gender'], gender['rec.gender'])
    print(gender_table)
    print(chi2_contingency(gender_table)[1])
    race_table = pd.crosstab(df['source.race'], df['rec.race'])
    print(chi2_contingency(race_table)[1])
    race = df[np.all([df['source.race'] != 'Unknown', df['rec.race'] != 'Unknown'], 0)]
    race['source.white'] = race['source.race'] == 'white'
    race['rec.white'] = race['rec.race'] == 'white'
    white_table = pd.crosstab(race['source.white'], race['rec.white'])
    print(white_table)
    print(chi2_contingency(white_table)[1])
    race['source.black'] = race['source.race'] == 'black or african american'
    race['rec.black'] = race['rec.race'] == 'black or african american'
    black_table = pd.crosstab(race['source.black'], race['rec.black'])
    print(black_table)
    print(chi2_contingency(black_table)[1])


def manual_override(df, username, gender = None, race = None):
    if gender:
        df.loc[df['source_username']==username, 'source.gender'] = gender
        df.loc[df['recommended_username']==username, 'rec.gender'] = gender
    if race:
        df.loc[df['source_username']==username, 'source.race'] = race
        df.loc[df['recommended_username']==username, 'rec.race'] = race

run_analysis(df)
manual_override(df, '@therock', 'Male')
manual_override(df, '@addisonre', 'Female', 'white')
manual_override(df, '@avani', 'Female', 'white')
manual_override(df, '@avneetkaur_13', 'Female', 'asian')
manual_override(df, '@babyariel', None, 'white')
manual_override(df, '@brentrivera', None, 'white')
manual_override(df, '@charlidamelio', 'Female', 'white')
manual_override(df, '@cznburak', None, 'middle eastern or north african')
manual_override(df, '@dixiedamelio', 'Female', 'white')
manual_override(df, '@dobretwins', None, 'white')
manual_override(df, '@gima_ashi', None, 'white')
manual_override(df, '@jacobsartorius', 'Male', 'white')
manual_override(df, '@jasoncoffee', 'Male', None)
manual_override(df, '@joealbanese', 'Male', 'white')
manual_override(df, '@justmaiko', 'Male', 'asian')
manual_override(df, '@kimberly.loaiza', None, 'hispanic, latino, or spanish origin')
manual_override(df, '@kristenhancher', None, 'white')
manual_override(df, '@laurengodwin', None, 'white')
manual_override(df, '@lizzza', 'Female', 'asian')
manual_override(df, '@lorengray', None, 'white')
manual_override(df, '@nishaguragain', None, 'asian')
manual_override(df, '@oye_indori', None, 'asian')
manual_override(df, '@piyanka_mongia', None, 'asian')
manual_override(df, '@sameeksha_sud', None, 'asian')
manual_override(df, '@savv.labrant', None, 'white')
manual_override(df, '@selenagomez', 'Female', 'hispanic, latino, or spanish origin')
manual_override(df, '@stokestwins', 'Male', 'white')
manual_override(df, '@theshilpashetty', None, 'asian')
manual_override(df, '@zachking', None, 'asian')
manual_override(df, '@tonylopez', 'Male', None)
manual_override(df, '@daviddobrik', None, 'white')
manual_override(df, '@lilhuddy', 'Male', 'white')
run_analysis(df)

def representation(df):
    race = df[np.all([df['source.race'] != 'Unknown', df['rec.race'] != 'Unknown'], 0)]
    race_table = pd.crosstab(df['source.race'], df['rec.race'])
    source_rep = sum(race_table['white'])
    rec_rep = sum(race_table.loc['white'])
    total = np.sum(np.sum(race_table))
    print(f'Source: {source_rep/total}. Rec: {rec_rep/total}')
representation(df)