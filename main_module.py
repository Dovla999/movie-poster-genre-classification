import glob

import pandas as pd
import numpy as np
import io
import sys
import os
import urllib.request

from tqdm import tqdm


def download_images():
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
    df.to_csv("MovieGenreDownload.csv", index=None)


def cleanup_data(df):
    image_paths = []
    imdb_id = []
    genres = []
    titles = []

    for file in glob.glob("Posters/*.jpg"):
        try:
            img_id = file[file.find('\\') + 1: file.find('.')]
            title = df[df["imdbId"] == int(img_id)]["Title"].values[0]
            genre = df[df["imdbId"] == int(img_id)]["Genre"].values[0]
            if genre == "": continue

            image_paths.append(file)
            imdb_id.append(img_id)
            genres.append(genre)
            titles.append(title)
        except:
            pass

    df = pd.DataFrame({'Image_Paths': image_paths, 'imdbId': imdb_id, 'Genre': genres, 'Title': titles})
    df.to_csv("MovieGenreProc.csv", index=None)


def to_bit_words(df):
    genres_all = df['Genre']
    genres_set = []
    for genres in genres_all:
        try:
            for genre in genres.split('|'):
                if genre not in genres_set: genres_set.append(genre)
        except:
            pass
    genres_list = list(genres_set)

    def convert(img_id, genres):
        genres_split = genres.split('|')
        row = [img_id]

        for i in range(len(genres_list)):
            if genres_list[i] in genres_split:
                row.append(1)
            else:
                row.append(0)
        row.append(genres)
        return row

    data = []
    for index, row in tqdm(df.iterrows()):
        try:
            data.append(convert(row['Image_Paths'], row['Genre']))
        except:
            pass
    np.savetxt("Converted_data.csv", np.asarray(data), fmt='%s', delimiter=" ")


def generate_sets():
    df = pd.read_csv("Converted_data.csv", delimiter=" ")
    random_seed = 50
    train_df = df.sample(frac=0.70, random_state=random_seed)
    tmp_df = df.drop(train_df.index)
    test_df = tmp_df.sample(frac=0.33, random_state=random_seed)
    valid_df = tmp_df.drop(test_df.index)

    print("Train_df=", len(train_df))
    print("Val_df=", len(valid_df))
    print("Test_df=", len(test_df))

    np.savetxt("Train.csv", train_df, fmt='%s', delimiter=" ")
    np.savetxt("Test.csv", test_df, fmt='%s', delimiter=" ")
    np.savetxt("Valid.csv", valid_df, fmt='%s', delimiter=" ")

def main():
    if '--download' in sys.argv:
        pass
        # download_images()
    # df = pd.read_csv("MovieGenreDownload.csv", encoding='ISO-8859-1')
    # cleanup_data(df)
    #df = pd.read_csv("MovieGenreProc.csv", encoding='ISO-8859-1')
    #to_bit_words(df)
    #generate_sets()

if __name__ == "__main__":
    main()
