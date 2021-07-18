# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 19:19:15 2021

@author: alekp
"""

from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from autokeras import StructuredDataClassifier
import autokeras as ak


def autokerasModel(X, Y, validation_split=0.15, epochs=50):
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # Initialize the image classifier.
    searchmodel = ak.ImageClassifier(
        overwrite=True,
        max_trials=1)
    
    # Feed the image classifier with training data.
    searchmodel.fit(X, Y,
                    validation_split=validation_split,
                    epochs=epochs)
    
    return searchmodel
    
    # print("Red buscada")
    # model = searchmodel.export_model()
    
    # print(model.summary())
    
    # # Predict with the best model.
    # predicted_y = model.predict(X_test)
    # print(predicted_y)
    
    # # Evaluate the best model with testing data.
    # print(searchmodel.evaluate(X_test, y_test))