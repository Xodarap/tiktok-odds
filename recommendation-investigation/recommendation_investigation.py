# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:25:23 2020

@author: bwsit
"""


import psycopg2
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

conn=psycopg2.connect('dbname=postgres user=postgres')
cur = conn.cursor()

cur.execute("""        
select source_username, source.gender, source.race,
    recommended_username, rec.gender, rec.race
    from tiktok.race_check
    inner join tiktok.clarifai_data source on source.username = source_username
    inner join tiktok.clarifai_data rec on rec.username = recommended_username
    where tiktok.race_check.old is null
""")

res=cur.fetchall()
conn.close()

df = pd.DataFrame(res, columns = ['source_username', 'source.gender', 'source.race',
    'recommended_username', 'rec.gender', 'rec.race'])

def run_analysis(df):
    gender = df[np.all([df['source.gender'] != 'Unknown', df['rec.gender'] != 'Unknown'], 0)]
    gender_table = pd.crosstab(gender['source.gender'], gender['rec.gender'])
    print(gender_table)
    print(chi2_contingency(gender_table)[1])
    race = df.loc[np.all([df['source.race'] != 'Unknown', df['rec.race'] != 'Unknown'], 0)]
    race_table = pd.crosstab(race['source.race'], race['rec.race'])
    print(race_table)
    print(chi2_contingency(race_table)[1])
    race = df[np.all([df['source.race'] != 'Unknown', df['rec.race'] != 'Unknown'], 0)]
    race.loc[:, 'source.white'] = race['source.race'] == 'white'
    race.loc[:, 'rec.white'] = race['rec.race'] == 'white'
    white_table = pd.crosstab(race['source.white'], race['rec.white'])
    print(white_table)
    print(chi2_contingency(white_table)[1])
    race.loc[:, 'source.black'] = race['source.race'] == 'black or african american'
    race.loc[:, 'rec.black'] = race['rec.race'] == 'black or african american'
    black_table = pd.crosstab(race['source.black'], race['rec.black'])
    print(black_table)
    print(chi2_contingency(black_table)[1])
    return (gender_table, race_table)

run_analysis(df)

def manual_override(df, username, gender = None, race = None):
    if gender:
        df.loc[df['source_username']==username, 'source.gender'] = gender
        df.loc[df['recommended_username']==username, 'rec.gender'] = gender
    if race:
        df.loc[df['source_username']==username, 'source.race'] = race
        df.loc[df['recommended_username']==username, 'rec.race'] = race

overrides = [['@therock', 'Male', None],
    ['@addisonre', 'Female', 'white'],
    ['@avani', 'Female', 'white'],
    ['@avneetkaur_13', 'Female', 'asian'],
    ['@babyariel', None, 'white'],
    ['@brentrivera', None, 'white'],
    ['@charlidamelio', 'Female', 'white'],
    ['@cznburak', None, 'middle eastern or north african'],
    ['@dixiedamelio', 'Female', 'white'],
    ['@dobretwins', None, 'white'],
    ['@gima_ashi', None, 'white'],
    ['@jacobsartorius', 'Male', 'white'],
    ['@jasoncoffee', 'Male', None],
    ['@joealbanese', 'Male', 'white'],
    ['@justmaiko', 'Male', 'asian'],
    ['@kimberly.loaiza', None, 'hispanic, latino, or spanish origin'],
    ['@kristenhancher', None, 'white'],
    ['@laurengodwin', None, 'white'],
    ['@lizzza', 'Female', 'asian'],
    ['@lorengray', None, 'white'],
    ['@nishaguragain', None, 'asian'],
    ['@oye_indori', None, 'asian'],
    ['@piyanka_mongia', None, 'asian'],
    ['@sameeksha_sud', None, 'asian'],
    ['@savv.labrant', None, 'white'],
    ['@selenagomez', 'Female', 'hispanic, latino, or spanish origin'],
    ['@stokestwins', 'Male', 'white'],
    ['@theshilpashetty', None, 'asian'],
    ['@zachking', None, 'asian'],
    ['@tonylopez', 'Male', None],
    ['@daviddobrik', None, 'white'],
    ['@lilhuddy', 'Male', 'white'],
    ['@chrisbrownofficial', 'Male', 'black or african american'],
    ['@brunomars', None, 'black or african american'],
    ['@xtina', 'Female', 'hispanic, latino, or spanish origin'],
    ['@daddyyankee', None, 'hispanic, latino, or spanish origin'],
    ['@camilacabello', 'Female', 'hispanic, latino, or spanish origin'],
    ['@shakira', None, 'hispanic, latino, or spanish origin'],
    ['@tyga', 'Male', 'asian'],
    ['@tyrabanks', 'Female', 'black or african american'],
    ['@mileycyrus', 'Female', None],
    ['@katyperry', None, 'white'],
    ['@dualipaofficial', None, 'white'],
    ['@postmalone', None, 'white'],
    ['@maroon5', 'Male', 'white'],
    ['@halsey', None, 'black or african american'],
    ['@mariahcarey', None, 'black or african american'],
    ['@dojacat', None, 'black or african american'],
    ['@lizzo', 'Female', 'black or african american'],
    ['@marshmellomusic', 'Male', 'white'],
    ['@21savage', 'Male', 'black or african american'],
    ['@alesso', 'Male', 'white'],
    ['@anitta', 'Female', 'hispanic, latino, or spanish origin'],
    ['@iamcardib', 'Female', 'black or african american'],
    ['@jonasbrothers', 'Male', 'white'],
    ['@ashleytisdale', 'Female', 'white'],
    ['@avamax', 'Female', 'white'],
    ['@bazziofficial', 'Male', 'middle eastern or north african'],
    ['@brycehall', 'Male', 'white'],
    ['@camilo', 'Male', 'hispanic, latino, or spanish origin'],
    ['@iambeckyg', 'Female', 'hispanic, latino, or spanish origin'],
    ['@kygomusic', 'Male', 'white'],
    ['@lauvsongs', 'Male', 'white'],
    ['@lilnasx', 'Male', 'black or african american'],
    ['@papijuancho', 'Male', 'hispanic, latino, or spanish origin'],
    ['@sia', 'Female', 'white'],
    ['@panicatthedisco', 'Male', 'white'],
    ['@thechainsmokers', 'Male', 'white'],
    ['@zoelaverne', 'Female', 'white']]
for override in overrides:
    manual_override(df, *override)
gender_table, race_table = run_analysis(df)

def representation(df):
    race = df.loc[np.all([df['source.race'] != 'Unknown', df['rec.race'] != 'Unknown'], 0)]
    race_table = pd.crosstab(race['source.race'], race['rec.race'])
    saved = race
    info = []
    total = np.sum(np.sum(race_table))
    all_races = np.unique(np.append(race_table.index.values, race_table.columns.values))
    for race in all_races:
        if race in race_table.columns:
            rec_rep = sum(race_table[race])
        else:
            rec_rep = 0
        source_rep = sum(race_table.loc[race])
        info.append([race, source_rep, source_rep/total, rec_rep, rec_rep / total])
    info_df = pd.DataFrame(info, columns = ['Race', 'Source #', 'Source %', 'Rec #', 'Rec %'])
    print(info_df)
    con = info_df[['Source #', 'Rec #']]
    print(chi2_contingency(con)[1])
    return saved
race_representation = representation(df)

gender_table.to_csv('gender.csv')
race_table.to_csv('race.csv')
race_representation.to_csv('race_representation.csv')
df.to_csv('raw_data.csv')