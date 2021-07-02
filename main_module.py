import glob

import pandas as pd
import numpy as np
import io
import sys
import os
import urllib.request

from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from keras import models
from keras import layers
from keras import optimizers
from PIL import Image
from statistics import mean


def download_images():
    df = pd.read_csv("MovieGenre.csv", encoding='ISO-8859-1')
    df.dropna(how='any')
    invalid = []
    for index, row in df.iterrows():

        if not os.path.exists('Posters'):
            os.makedirs('Posters')

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
    for index, row in df.iterrows():
        try:
            data.append(convert(row['Image_Paths'], row['Genre']))
        except:
            pass
    np.savetxt("Converted_data.csv", np.asarray(data), fmt='%s', delimiter=" ")


def generate_sets():
    df = pd.read_csv("Converted_data.csv", delimiter=" ")
    random_seed = 100
    train_df = df.sample(frac=0.80, random_state=random_seed)
    tmp_df = df.drop(train_df.index)
    test_df = tmp_df.sample(frac=0.5, random_state=random_seed)
    valid_df = tmp_df.drop(test_df.index)

    print("Train_df=", len(train_df))
    print("Val_df=", len(valid_df))
    print("Test_df=", len(test_df))

    np.savetxt("Train.csv", train_df, fmt='%s', delimiter=" ")
    np.savetxt("Test.csv", test_df, fmt='%s', delimiter=" ")
    np.savetxt("Valid.csv", valid_df, fmt='%s', delimiter=" ")


def load_images(df):
    data = []
    paths = np.asarray(df.iloc[:, 0])

    for item in tqdm(range(len(paths))):
        img = image.load_img(paths[item], target_size=(200,150,3))
        img=image.img_to_array(img)
        img = img/255
        data.append(img)

    return np.array(data), np.array(df.iloc[:, 1:29])



def load_images_gen(df, size):
    paths = np.asarray(df.iloc[:, 0])
    l = len(paths)-5



    while True:
        batch_start = 0
        batch_end = size

        while batch_start < l:
            limit = min(batch_end, l)
            data = []

            for item in tqdm((range(batch_start, limit))):
                img = image.load_img(paths[item], target_size=(200, 150, 3))
                img = image.img_to_array(img)
                img = img / 255
                data.append(img)
            batch_start += size
            batch_end += size
        yield np.array(data), np.array(df.iloc[:, 1:29])


def train_first():
    df_t = pd.read_csv("Train.csv", delimiter=" ")
    x_train, y_train = load_images(df_t)

    df_v = pd.read_csv("Valid.csv", delimiter=" ")
    x_val, y_val = load_images(df_v)

    num_classes = 28

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(200, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), batch_size=2)
    model.save('first_model')


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def main():
    if '--download' in sys.argv:
        pass
        # download_images()
    # df = pd.read_csv("MovieGenreDownload.csv", encoding='ISO-8859-1')
    # cleanup_data(df)
    # df = pd.read_csv("MovieGenreProc.csv", encoding='ISO-8859-1')
    # to_bit_words(df)
    #generate_sets()
    train_first()


if __name__ == "__main__":
    main()
