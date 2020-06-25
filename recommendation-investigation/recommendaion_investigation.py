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
    return race_table


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
manual_override(df, '@chrisbrownofficial', 'Male', 'black or african american')
manual_override(df, '@brunomars', None, 'black or african american')
manual_override(df, '@xtina', 'Female', 'hispanic, latino, or spanish origin')
manual_override(df, '@daddyyankee', None, 'hispanic, latino, or spanish origin')
manual_override(df, '@camilacabello', 'Female', 'hispanic, latino, or spanish origin')
manual_override(df, '@shakira', None, 'hispanic, latino, or spanish origin')
manual_override(df, '@tyga', 'Male', 'asian')
manual_override(df, '@tyrabanks', 'Female', 'black or african american')
manual_override(df, '@mileycyrus', 'Female', None)
manual_override(df, '@katyperry', None, 'white')
manual_override(df, '@dualipaofficial', None, 'white')
manual_override(df, '@postmalone', None, 'white')
manual_override(df, '@maroon5', 'Male', 'white')
manual_override(df, '@halsey', None, 'black or african american')
manual_override(df, '@mariahcarey', None, 'black or african american')
manual_override(df, '@dojacat', None, 'black or african american')
manual_override(df, '@lizzo', 'Female', 'black or african american')
manual_override(df, '@marshmellomusic', 'Male', 'white')
manual_override(df, '@21savage', 'Male', 'black or african american')
manual_override(df, '@alesso', 'Male', 'white')
manual_override(df, '@anitta', 'Female', 'hispanic, latino, or spanish origin')
manual_override(df, '@iamcardib', 'Female', 'black or african american')
manual_override(df, '@jonasbrothers', 'Male', 'white')
manual_override(df, '@ashleytisdale', 'Female', 'white')
manual_override(df, '@avamax', 'Female', 'white')
manual_override(df, '@bazziofficial', 'Male', 'middle eastern or north african')
manual_override(df, '@brycehall', 'Male', 'white')
manual_override(df, '@camilo', 'Male', 'hispanic, latino, or spanish origin')
manual_override(df, '@iambeckyg', 'Female', 'hispanic, latino, or spanish origin')
manual_override(df, '@kygomusic', 'Male', 'white')
manual_override(df, '@lauvsongs', 'Male', 'white')
manual_override(df, '@lilnasx', 'Male', 'black or african american')
manual_override(df, '@papijuancho', 'Male', 'hispanic, latino, or spanish origin')
manual_override(df, '@sia', 'Female', 'white')
manual_override(df, '@panicatthedisco', 'Male', 'white')
manual_override(df, '@thechainsmokers', 'Male', 'white')
manual_override(df, '@zoelaverne', 'Female', 'white')
race_table = run_analysis(df)

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
    #print(f'Source: {source_rep/total}. Rec: {rec_rep/total}')
    print(info_df)
    con = info_df[['Source #', 'Rec #']]
    print(chi2_contingency(con)[1])
    return saved
rep = representation(df)
conn.close()