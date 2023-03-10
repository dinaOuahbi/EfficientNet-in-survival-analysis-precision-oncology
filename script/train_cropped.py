# python 3
# DO

'''
this script will use Efficient Net CNN model performed with random center cropping to train datasets

    - RCC will concerve the descriminatives caracteristics of the central region (img) that contain generaly more tumor cells
    - RCC will reduce under sampling scal


python script.py [E / R] [crop / trad]
'''


from tensorflow.keras.applications import * #Efficient Net included here
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import pandas as pd
from collections import Counter


DATASET_PATH  = '../TCGA_annot_DB10/'
IMAGE_SIZE    = (402, 402)
CROP_LENGTH   = 224
NUM_CLASSES   = 3
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20
WEIGHTS_FINAL = '../results/StromaVSautre_DB10_boosted_model/'

arg = sys.argv[1]
aug = sys.argv[2]

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


'''TRAIN'''
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


train_batches = train_datagen.flow_from_directory(DATASET_PATH + 'Train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)


'''VALIDATION'''
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + 'Val',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)




# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')


'''
#build our classifier model based on pre-trained EfficientNetB0:
    # 1. we don't include the top (fully connected) layers of EfficientNetB0
    # 2. we add a DropOut layer followed by a Dense (fully connected) layer which generates softmax class score for each class
    # 3. we compile the final model using an Adam optimizer, with a low learning rate (since we are 'fine-tuning')

'''
if aug == 'crop':
    input_shape = (CROP_LENGTH,CROP_LENGTH,3)
elif aug == 'trad':
    input_shape = IMAGE_SIZE+(3,)

if arg == 'E':
    model =  EfficientNetB0(include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape)
elif arg == 'R':
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=input_shape)


x = model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
model_final = Model(inputs=model.input, outputs=output_layer)


for layer in model_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
model_final.compile(optimizer=Adam(lr=0.00001),loss='categorical_crossentropy', metrics=['accuracy'])

print(model_final.summary())

# train the model
counter = Counter(train_batches.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

if aug == 'crop':

    train_crops = crop_generator(train_batches, CROP_LENGTH)
    valid_crops = crop_generator(valid_batches, CROP_LENGTH)

    history = model_final.fit_generator(train_crops,
                            class_weight = class_weights,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_crops,
                            validation_steps = valid_batches.samples // BATCH_SIZE,
                            epochs = NUM_EPOCHS)


elif aug == 'trad':
    history = model_final.fit_generator(train_batches,
                            class_weight = class_weights,
                            steps_per_epoch = train_batches.samples // BATCH_SIZE,
                            validation_data = valid_batches,
                            validation_steps = valid_batches.samples // BATCH_SIZE,
                            epochs = NUM_EPOCHS)
# save trained weights
model_final.save(f'{WEIGHTS_FINAL}{arg}_{aug}_cropped_final.h5')

# save model history
hist_df = pd.DataFrame(history.history)
hist_csv_file = f'{WEIGHTS_FINAL}{arg}_{aug}_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
