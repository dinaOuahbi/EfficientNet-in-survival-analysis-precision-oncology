#!/usr/bin/env python3.7

import getopt
import numpy as np
import keras
import sys
#from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img, smart_resize
from tensorflow.keras.utils import load_img, img_to_array, array_to_img, save_img
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import os
import pandas as pd
from random import *
import re
import seaborn as sns
import math
import tensorflow as tf
import shutil
import PIL


arg = sys.argv[1] # class


root_input = f'/work/shared/ptbc/CNN_Pancreas_V2/Donnees/project_kernel03_scan21000500/lame francois/bsc_class_classification_project/data/{arg}/'
files = os.listdir(root_input)

try:
    os.mkdir(f'../BSC_annot_DB11/{arg}')
except:
    pass

for file in files:
   print(file) # tile
   if (file.endswith(".jpg")):
       print("ok")
   # On charge l'image target : celle sur qui on veut normaliser toutes les autres
       img = load_img(f"../target.tif",
           target_size=(402, 402))
       img = img_to_array(img)
       img = img.astype(np.uint8)
       LAB = img/255
       Mu = LAB.sum(axis=0).sum(axis=0) / (LAB.size / 3)
       LAB[:, :, 0] = LAB[:, :, 0] - Mu[0]
       LAB[:, :, 1] = LAB[:, :, 1] - Mu[1]
       LAB[:, :, 2] = LAB[:, :, 2] - Mu[2]
       Sigma = ((LAB * LAB).sum(axis=0).sum(axis=0) / (LAB.size / 3 - 1)) ** 0.5
       target_mu = Mu
       target_sigma = Sigma

       # On charge l'image que l'on souhaite normaliser
       img_scr = load_img(os.path.join(root_input, file),target_size=(220, 220)).resize((402, 402))
       # get input image dimensions
       m = 402
       n = 402
       img_scr = img_to_array(img_scr)
       img_scr = img_scr.astype(np.uint8)
       compteur = 0
       # Reperage des images blanches
       for i in range(402):
           for j in range(402):
               if(img_scr[i,j,0]>210 and img_scr[i,j,1]>210 and img_scr[i,j,2]>210):
                   compteur += 1
       if (compteur > 2*402*402/3): #si l'image contient 2/3 de blanc (gris clair)
           root_output = f"../BSC_annot_DB11/drop/"
       else :
           root_output = f"../BSC_annot_DB11/{arg}/"
       im_lab = img_scr/255
       # Normalisation
       src_mu = None
       src_sigma = None
       # calculate src_mu if not provided
       if src_mu is None:
           src_mu = im_lab.sum(axis=0).sum(axis=0) / (m * n)
       # center to zero-mean
       for i in range(3):
           im_lab[:, :, i] = im_lab[:, :, i] - src_mu[i]
       # calculate src_sigma if not provided
       if src_sigma is None:
           src_sigma = ((im_lab * im_lab).sum(axis=0).sum(axis=0) / (m * n - 1)) ** 0.5
       # scale to unit variance
       for i in range(3):
           im_lab[:, :, i] = im_lab[:, :, i] / src_sigma[i]
       # rescale and recenter to match target statistics
       for i in range(3):
           im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]
       im_lab[im_lab > 1] = 1
       im_lab[im_lab < 0] = 0
       im_normalized = im_lab * 255
       #im_normalized = im_normalized.astype(np.uint8)
       im_normalized_image = array_to_img(im_normalized)
       save_img(os.path.join(root_output, file),im_normalized_image)
       #os.remove(os.path.join(root_input, file)) # pour ne pas avoir trop de fichiers on supprime les images brutes
