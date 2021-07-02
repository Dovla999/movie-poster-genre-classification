import glob

import pandas as pd
import numpy as nmp
import io
import sys
import os
import urllib.request

from tqdm import tqdm

df = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
df.dropna(how='any')

invalid = []
for index, row in tqdm(df.iterrows()):

    file_path = "Posters/" + str(row['imdbId']) + ".jpg"

    try:
        res = urllib.request.urlopen(row['Poster'])
        data = res.read()
        file = open(file_path, 'wb')
        file.write(bytearray(data))
        file.close()
    except:
        invalid.append(row['imdbId'])

df = df[~df['imdbId'].isin(invalid)]
df = df[['imdbId', 'Title', 'Genre']]

images = [file for file in glob.glob("Posters/*jpg")]

df.to_csv("MovieGenreProc.csv", index=None)
