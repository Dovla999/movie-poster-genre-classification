import glob
from random import randint

import pandas as pd
import numpy as np
import io
import sys
import os
import urllib.request

import tf
from keras.utils.data_utils import Sequence
from tqdm import tqdm

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

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

            image_paths.append('Posters/'+img_id+'.jpg')
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
    df = np.array_split(df, 1)[0]
    random_seed = 100
    train_df = df.sample(frac=0.70, random_state=random_seed)
    tmp_df = df.drop(train_df.index)
    test_df = tmp_df.sample(frac=1 / 3, random_state=random_seed)
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
        img = image.load_img(paths[item], target_size=(100, 75, 3))
        img = image.img_to_array(img)
        img = img / 255
        data.append(img)

    return np.array(data), np.array(df.iloc[:, 1:29])


class MyGenerator(Sequence):
    def __init__(self, my_set, batch_size):
        self.my_set = my_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.my_set) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.my_set[idx * self.batch_size:(idx + 1) * self.batch_size]
        return load_images(batch)


def train_first():
    df_t = pd.read_csv("Train.csv", delimiter=" ")

    train_gen = MyGenerator(df_t, 32)

    num_classes = 28

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=(100, 75, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    df_v = pd.read_csv("Valid.csv", delimiter=" ")
    x_val, y_val = load_images(df_v)

    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_gen, epochs=10, validation_data=(x_val, y_val))
    model.save('first_model')


def train_second():
    num_classes = 28

    df_t = pd.read_csv("Train.csv", delimiter=" ")

    train_gen = MyGenerator(df_t, 32)

    df_v = pd.read_csv("Valid.csv", delimiter=" ")
    x_val, y_val = load_images(df_v)

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(100, 75, 3))

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    model = models.Sequential()

    model.add(vgg_conv)

    num_classes = 28

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_gen, epochs=5, validation_data=(x_val, y_val))

    model.save('second_model')


def accuracy_score(test_path, model_path):
    test_df = pd.read_csv(test_path, delimiter=" ")
    X_test, Y_test = load_images(test_df)

    model = load_model(model_path)

    pred = model.predict(np.array(X_test))

    count = 0
    for i in tqdm(range(len(pred))):
        value = 0

        first3 = np.argsort(pred[i])[-3:]
        correct = np.where(Y_test[i] == 1)[0]

        for j in first3:
            if j in correct:
                value += 1

        if (value > 0):
            count = count + 1

    print("Correct = ", count)
    print("Total = ", len(pred))
    print("Accuracy = ", count / len(pred))


def test_singular(image_path, model_path):
    genres = ['Animation', 'Action', 'Adventure', 'Fantasy', 'Comedy', 'Drama', 'Crime', 'History', 'Romance', 'Family',
              'Thriller', 'Mystery', 'Documentary', 'Horror', 'Musical', 'War', 'Sci-Fi', 'Music', 'Biography', 'Sport',
              'Short', 'Western', 'Adult', 'Talk-Show', 'News', 'Reality-TV', 'Film-Noir', 'Game-Show']
    model = load_model(model_path)
    img = image.load_img(image_path, target_size=(100, 75, 3))
    img = image.img_to_array(img)
    img = img / 255
    prob = model.predict(img.reshape(1, 100, 75, 3))

    print((prob[0]))
    print(len(genres))
    top_3 = {genres[i]: prob[0][i] for i in range(0, 27)}

    top_3_genres = sorted(top_3, key=top_3.get, reverse=True)[:3]

    for genre in top_3_genres:
        print(f'{genre}: {top_3[genre]}')


def main():
    #download_images()
    #df = pd.read_csv("./MovieGenreDownload.csv", encoding='ISO-8859-1')
    #cleanup_data(df)
    #df = pd.read_csv("MovieGenreProc.csv", encoding='ISO-8859-1')
    #to_bit_words(df)
    #generate_sets()
    #train_first()
    #train_second()
    accuracy_score("Test.csv", "first_model")
    accuracy_score("Test.csv", "second_model")
    #test_singular(img_path,'first_model')




if __name__ == "__main__":
    main()
