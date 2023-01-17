#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from random import shuffle
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from random import sample
import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow import keras
import matplotlib.cm as cm

# In[77]:


def label_image_one_hot_encoder(img):
    ## Helper for process_data
    label = img.split('/')[3]
    if label == 'Duodenum': return 0
    elif label == 'N_T': return 1
    elif label == 'Stroma': return 2


# In[78]:


def process_data(image_list, DATA_FOLDER, IMG_SIZE):
  ## Helper for manual_pre_process
  ## Creates an array of images, labels, and file path
  ## Shuffles the array before returning
    data_df = []
    for img in tqdm(image_list):
        path = os.path.join(DATA_FOLDER, img)
        label = label_image_one_hot_encoder(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img), np.array(label), path])
    shuffle(data_df)
    return data_df


# In[79]:


def manual_pre_process(dir, IMG_SIZE, how_much_img):
    image_lst = sample(os.listdir(dir), how_much_img)
    data_df = process_data(image_lst, dir, IMG_SIZE)
    X = np.array([i[0] for i in data_df]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array([i[1] for i in data_df])
    files = np.array([i[2] for i in data_df])
    return X, y, files


# In[80]:



def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# In[67]:


# On charge le modele permettant de predire la duo et N_T et stroma
model = tf.keras.models.load_model('../results/StromaVSautre_DB10/StromaVSautre_efficientNet',compile=False)


# In[81]:


N_T_imgs = '../TCGA_annot_DB10/Test/N_T/'
Duodenum_imgs = '../TCGA_annot_DB10/Test/Duodenum/'
Stroma_imgs = '../TCGA_annot_DB10/Test/Stroma/'

decode_predition = {0:'Duodenum',1:'N_T',2:'Stroma'}

X_NT, y_NT, files_NT = manual_pre_process(N_T_imgs, 402, 100)
X_Duodenum, y_Duodenum, files_Duodenum = manual_pre_process(Duodenum_imgs, 402, 100)
X_Stroma, y_Stroma, files_Stroma = manual_pre_process(Stroma_imgs, 402, 100)


# In[131]:


def predict_img(X, y, files, plot=True, sample=10):
    plt.figure(figsize=(20, 8))
    cpt = 0
    for i in range(sample):
        img = X[i]
        label = y[i]
        path = files[i]
        argmax_prob = tf.math.argmax(model.predict(np.expand_dims(img, axis=0))[0]).numpy()
        if label == argmax_prob:
            cpt+=1
        pred_raw = max(model.predict(np.expand_dims(img, axis=0))[0])
        pred_label = decode_predition[argmax_prob]
        if plot:
            ax = plt.subplot(sample*2//10, 5, i+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(decode_predition[label] + ' | ' + pred_label + ' : ' + str(pred_raw))

    return cpt/sample


# # GRAD CAM


# In[160]:


size = 402
def build_model_gradcam(size):
    base = EfficientNetB0(input_shape= (size,size,3),
                                      include_top=False,
                                      weights='imagenet')
    x= GlobalAveragePooling2D()(base.output)
    x=Dense(3,activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    opt = tf.optimizers.SGD(learning_rate=1e-3)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,loss=loss,metrics=["AUC"])
    return model




# In[164]:


grad_model = build_model_gradcam(size)
grad_model.set_weights(model.get_weights())


# In[167]:


from tensorflow.keras.models import Model
def GradCam(model, img_array, layer_name, eps=1e-8):
    gradModel = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output,
        model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, 0]

    print('convOutputs', convOutputs)
    print('predictions', predictions)
    grads = tape.gradient(loss, convOutputs)
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")

    guidedGrads = castConvOutputs * castGrads * grads

    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    weights = tf.reduce_mean(guidedGrads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom

    return heatmap

def display_gradcam(img, heatmap,alpha):
    # Load the original image
    img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


# In[168]:

img = X_Stroma[50]/255
grad_cam = GradCam(grad_model, np.expand_dims(img, axis=0), 'block5b_dwconv')
grad_cam_superimposed = display_gradcam(img, grad_cam, 0.05)
print(grad_cam)

plt.figure(figsize=(12, 5))
ax = plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')
ax = plt.subplot(1, 2, 2)
plt.imshow(grad_cam_superimposed)
plt.axis('off')
plt.title('top_activation Grad-CAM heat-map')
plt.tight_layout()

plt.savefig('grad_cam_superimposed.jpg')
keras.preprocessing.image.save_img('grad_cam_superimposed.jpg',grad_cam_superimposed)

# In[ ]:
