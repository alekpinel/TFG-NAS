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

from data_reading_visualization import extraerSP_SS, convertToBinary

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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

  #model.add(Dense(64, activation='relu'))
  #model.add(Dropout(0.5))

  model.add(Dense(output_size, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
  
  return model

def leaveOneOutBinary(X, y_no_categorical, epocas,tam_batch):
  mean = []
  loo = LeaveOneOut()
  loo.get_n_splits(X)
  y_predict = []
  
  y = convertToBinary(y_no_categorical)
  y = to_categorical(y)

  contador = 0
  for t_v_i, test_i in loo.split(X):
      print(color.BOLD + 'LOO '+ str(contador) + ":" + color.END)
      contador +=1

      model = OriginalNN((224,224,3), 2)
      # print(model.summary())

      X_train = X[t_v_i]
      y_train = y[t_v_i]

      X_test = X[test_i]
      y_test = y[test_i]
      h = model.fit(X_train, y_train, epochs = epocas ,batch_size= tam_batch, verbose= 0)
      print("Mean Loss ",round(statistics.mean(h.history['loss']),3))
      print("Mean Accuracy ",round(statistics.mean(h.history['accuracy']),3))

      y_pre = model.predict_classes(X_test)
      y_predict.append(y_pre)

  cm = metrics.confusion_matrix(y_no_categorical, y_predict)

  accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)

  return round(accuracy,3), round(specificity,3), round(sensitivity,3), round(precision,3), round(f1score,3)  , cm


def OriginalNN2(data_sets_names,data_sets, Y):
    df_salida_binary = pd.DataFrame(columns=["DS","ACC","Specificity","Sensitivity", "Precision", "Ponderacion"])
    df_salida_binary.head()
    
    count = 0
    matrices_cf_binary = []
    
    tam_batch = 1
    total_epocas = 10
    
    for name_ds, ds in zip(data_sets_names,data_sets):
      print(name_ds)
      #sc-> score sp->specificity ss->sensitivity
      sc, sp, ss, pr, f1, cm = leaveOneOutBinary(ds,Y, total_epocas, tam_batch)
    
      pond = 0.35 * ss + 0.25 * sc + 0.2 * sp + 0.2 * pr 
    
      df_salida_binary.loc[count] = [name_ds,   sc,   sp, ss, pr, pond]
      
      print(f"Results {name_ds}: sc: {sc} sp: {sp} ss: {ss} pr: {pr} pond: {pond}")
    
      count+=1
      matrices_cf_binary.append(cm)
      
    df_salida_binary.head()