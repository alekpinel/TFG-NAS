# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:25:49 2021

@author: alekp
"""

import math
import cv2
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.applications import ResNet50, VGG16
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import re
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut
import matplotlib.pyplot as plt
import random
import json
import statistics
from sklearn import metrics
import sys

from utils import extraerSP_SS, convertToBinary

def OriginalNN(input_size_net, output_size, X=None, Y=None):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_size_net))
  
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.5))

  model.add(Flatten())

  # model.add(Dense(output_size, activation='softmax'))
  model.add(Dense(output_size, activation='sigmoid'))
  

  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
  
  return model