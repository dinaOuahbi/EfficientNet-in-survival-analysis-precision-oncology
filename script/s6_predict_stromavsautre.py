#!/usr/bin/env python3.7
# -*-coding:Latin-1 -*

"""
# ---------------------------------------------
# Programme: predict_stromavsautre
# Auteur EB
# Date 15/11/2021
#
# Prediction a partir du CNN permettant la classification
# des tuiles Stromales vs Normales/Tumorales
# -------------------------------------------------------------
"""

# On importe tous les modules dont on a besoin
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical
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
tps1 = time.clock()
arg = sys.argv[1]
#21_0_005_00_13
#arg = 'TCGA-IB-AAUP-01Z-00-DX1.88461844-4FC6-4075-BE97-B5A239A3A260'
output = '../results/StromaVSautre_DB10/LamesCompletes/Stroma/'

# On charge le modele entraine
model = tf.keras.models.load_model('../results/StromaVSautre_DB10/StromaVSautre_efficientNet/', compile = False)

test_datagen = ImageDataGenerator()

batch_size = 32
# Générateurs
bsc_dir="/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/SCAN_reinhard/"

bsc_generator = test_datagen.flow_from_directory(
    directory=bsc_dir, classes = [arg],
    target_size =(402, 402),
    batch_size=batch_size,
    shuffle=False,
    seed=42
)

# On va enregistrer la matrice qui contient les probabilités
myName = bsc_generator.filenames
test_predictions_baseline = model.predict(bsc_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName)
myPrediction.to_csv(output+"myPrediction_"+arg+".csv", sep=',', encoding='utf-8',
               index=True, header = None)
