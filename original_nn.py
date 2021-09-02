# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:25:49 2021

@author: alekp
"""

import keras
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential



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