#!/usr/bin/env python3.7

"""
# ---------------------------------------------
# Programme: DenseNet.py
# Auteur EB
# Date 15/11/2021
#
# Prediction a partir du CNN permettant la classification
# des tuiles Tumorales vs Autres
# -------------------------------------------------------------
"""

# On importe tous les modules dont on a besoin
import warnings
import PIL
import time
import tensorflow as tf
import progressbar
from tensorflow.keras import datasets, layers, models
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras import optimizers
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import numpy as np
import os
import pandas as pd
import sys
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import math
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from sklearn.utils import class_weight
import pathlib
import functools
from keras.utils import np_utils
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import time
from datetime import datetime
#####################################################################

slide = sys.argv[1]
print(f'slide prediction  .............: {slide}')#
OUTPUT = f'../results/StromaVSautre_DB10/LamesCompletes/N_T/'
INPUT=f'../results/N_Tselect/'
#feature_path = f'{SCNN}Features/'
EfficientNet = tf.keras.models.load_model('../results/Normal_Tumor_DB11_gpu/Normal_Tumor_efficientNet/', compile = False)
test_datagen = ImageDataGenerator()
batch_size = 32


def to_float(df):
    for col in df.columns[1:]:
        df[col] = df[col].astype('float')


def get_argmax(df):
    preds = list()
    for i in range(len(df)):
        probs = np.array(df.iloc[i, 1:]).tolist()
        preds.append(tf.math.argmax(probs).numpy())

def predict(slide, model):
    BSC_generator = test_datagen.flow_from_directory(
        directory=INPUT, classes = [slide],
        target_size =(402, 402),
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    tile_names = BSC_generator.filenames
    # select f
    intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer("global_average_pooling2d").output)
    intermediate_output = intermediate_layer_model.predict(BSC_generator)
    myPrediction = pd.DataFrame(intermediate_output, index = tile_names)
    myPrediction.to_csv(f'../results/StromaVSautre_DB10/LamesCompletes/Features_intermediate_layer_model_NT/Features_{slide}.csv', sep=',', encoding='utf-8',index=True, header = None)

    # predict
    test_predictions_baseline = model.predict(BSC_generator)
    myPrediction = pd.DataFrame(test_predictions_baseline, index = tile_names)
    myPrediction.to_csv(f'{OUTPUT}myPrediction_{slide}.csv', sep=',', encoding='utf-8',index=True, header = None)

if __name__ == '__main__':
    predict(slide=slide, model = EfficientNet)

    ##
    df = pd.read_csv(f'{OUTPUT}myPrediction_{slide}.csv')
    df = df.T.reset_index(drop=True).T.rename(columns={0:'tiles', 1:'Normal', 2:'Tumor'})
    preds = list()
    for i in range(len(df)):
        preds.append(tf.math.argmax(np.array(df.iloc[i, 1:].tolist())).numpy())
    df['class'] = preds
    mydict = dict()
    mydict[slide] = df['class'].value_counts()

    file = open("../results/StromaVSautre_DB10/LamesCompletes/N_T_distribution.txt","a")
    for key, value in mydict.items():
        file.write('%s:%s\n' % (key, value))
    file.close()
