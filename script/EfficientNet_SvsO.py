
#!pip  install tensorflow-gpu
from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf #Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
from collections import Counter
from tensorflow.keras.models import Model


###################################################################################### CTE ############################################################################################
NUMBER_OF_CLASSES = 3
SIZE = 402
BATCH_SIZE = 32
EPOCHS=10
#NUMBER_OF_TRAINING_IMAGES = len(train_generator.classes)
#NUMBER_OF_VALIDATION_IMAGES = len(validation_generator.classes)
input_shape = (SIZE, SIZE, 3)


###################################################################################### TPU / CPU / GPU ############################################################################################
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

###################################################################################### MODEL ############################################################################################
resize_and_rescale = tf.keras.Sequential([
layers.experimental.preprocessing.Resizing(SIZE, SIZE),
tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

conv_base = EfficientNetB0(weights="imagenet",include_top=False, input_shape=input_shape) # transfert learning
conv_base.trainable = True

# Construction du modele : with this architectur i got a very underfitting model (acc training bigger than acc val)
#model = models.Sequential()
#model.add(conv_base)
#model.add(layers.GlobalMaxPooling2D(name="gap")) #GAP layers are used to reduce the spatial dimensions of a three-dimensional tensor.
#model.add(layers.Dropout(.2, name="dropout_out"))   # chaque neurone a une chance sur 20 d’être désactivé. ==> eviter l'overfiting / lors des test, les neurones se reactives et leurs poinds sont multiplier par 0.2
#model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))     # Set NUMBER_OF_CLASSES to the number of your final predictions.
#conv_base.trainable = False

# Construction du modele
inputs = tf.keras.Input(shape=(402, 402, 3))
x = resize_and_rescale(inputs)
x = conv_base(x, training=True)
x= layers.GlobalAveragePooling2D()(x)
outputs=layers.Dense(3,activation='softmax')(x)
model = Model(inputs, outputs)

###################################################################################### GENERATORS ############################################################################################
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255) # Note that the validation data should not be augmented!

###################################################################################### DATA ############################################################################################
train_generator = train_datagen.flow_from_directory(
    '../TCGA_annot_DB10/Train',# This is the target directory
    target_size=(SIZE, SIZE),# All images will be resized to target height and width.
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",# Since we use categorical_crossentropy loss, we need categorical labels
    shuffle=False,
    seed=42
)
validation_generator = test_datagen.flow_from_directory(
    '../TCGA_annot_DB10/Val',
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=42
)
###################################################################################### PREFETCH AUTOTUNE ############################################################################################
try:
    train_generator = train_generator.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_generator = validation_generator.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
except:
    pass
###################################################################################### COMPILE / weighted AND FIT ############################################################################################
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=tf.keras.optimizers.Adam(0.00001), metrics = ["accuracy"])

# On calcule le poids de chacune des trois classes
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}


# Entrainement
history = model.fit(train_generator,
                    epochs=EPOCHS,class_weight = class_weights,
                    validation_data=validation_generator, verbose = 2)

###################################################################################### SAVE MODEL ############################################################################################
model.save('../results/StromaVSautre_DB10/StromaVSautre_efficientNet')
model.save('../results/StromaVSautre_DB10/StromaVSautre_efficientNet_weight.h5')

hist_df = pd.DataFrame(history.history)
hist_csv_file = '../results/StromaVSautre_DB10/StromaVSautre_efficientNet_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


###################################################################################### GENERATORS FOR PREDICT STEP (WITHOUT DATA AUGMENTED) ############################################################################################
output = '../results/StromaVSautre_DB10/'

train_generator = test_datagen.flow_from_directory(
    directory='../TCGA_annot_DB10/Train',
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
val_generator = test_datagen.flow_from_directory(
    directory='../TCGA_annot_DB10/Val',
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory='../TCGA_annot_DB10/Test',
    target_size=(SIZE, SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)
###################################################################################### PREFETCH AUTOTUNE ############################################################################################
try:
    train_generator = train_generator.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_generator = validation_generator.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_generator = test_generator.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
except:
    pass
###################################################################################### PREDICT ############################################################################################
myName = train_generator.filenames
col = train_generator.class_indices.keys()
test_predictions_baseline = model.predict(train_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName, columns=col)
myPrediction.to_csv(output+"myPrediction_train.csv", sep=',', encoding='utf-8',
               index=True, header = None)

myName = val_generator.filenames
col = val_generator.class_indices.keys()
test_predictions_baseline = model.predict(val_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName, columns=col)
myPrediction.to_csv(output+"myPrediction_val.csv", sep=',', encoding='utf-8',
               index=True, header = None)

myName = test_generator.filenames
col = test_generator.class_indices.keys()
test_predictions_baseline = model.predict(test_generator)
myPrediction = test_predictions_baseline
myPrediction = pd.DataFrame(myPrediction, index = myName, columns=col)
myPrediction.to_csv(output+"myPrediction_test.csv", sep=',', encoding='utf-8',
               index=True, header = None)
