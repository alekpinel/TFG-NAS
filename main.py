# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:49:23 2020

@author: alekp
"""

import math
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut
from numpy.random import seed
import time
from tensorflow.keras.utils import to_categorical
import statistics
from sklearn import metrics

from data_reading_visualization import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, convertToBinary
from original_nn import autokerasModel
from autokeras_model import createModelMnist


def NASExperiment(X, Y, model_name, NAS_function, NAS_parameters, test_percent=0.3, epochs=50, batch_size=32, verbose=1, save_results=True):
    
    # Split the data and prepare the binary
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percent, stratify=Y)
    Y_train = convertToBinary(Y_train)
    Y_test = convertToBinary(Y_test)
    
    
    # Apply NAS to search the architecture
    start_time = time.time()
     
    NAS_model = NAS_function(**NAS_parameters)
    
    end_time = time.time()
    seconds = end_time - start_time
    searching_time = seconds
    
    
def leaveOneOut(X_train, X_test, Y_train, Y_test, model, epochs=50, batch_size=32, verbose=1): 
    loo = LeaveOneOut()
    loo.get_n_splits(X_test)
    y_predict = []
    times = []
    
    labels_train = Y_train
    labels_test = Y_test
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
  
    contador = 0
    for t_v_i, test_i in loo.split(X_test):
        if (verbose):
            print(f"LOOP {contador}:")
       
        X_fold_train = np.concatenate(X_train, X_test[t_v_i])
        Y_fold_train = y[t_v_i]
  
        X_fold_test = X_test[test_i]
        Y_fold_test = Y_test[test_i]
        
        start_time = time.time()
        
        model = model_function(**model_parameters)
        
        
        h = model.fit(X_train, y_train, epochs=epochs ,batch_size=batch_size, verbose=0)
          
        end_time = time.time()
        seconds = end_time - start_time
        times.append(seconds)
        
        if (verbose):
            print("Train Loss ",round(statistics.mean(h.history['loss']),3))
            print("Train Accuracy ",round(statistics.mean(h.history['accuracy']),3))
  
        y_pre = model(X_test)
        y_pre = [np.argmax(values) for values in y_pre]
        # print(f"Y_true: {np.argmax(y_test, axis=1)} Y_pre: {y_pre}")
        y_predict.append(y_pre)
        
        if (verbose):
            print(f"Eval Accuracy: {CalculateAccuracy(np.argmax(y_test, axis=1), y_pre)}")
        
        contador +=1
  
    cm = metrics.confusion_matrix(labels, y_predict)
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    mean_time = sum(times) / len(times)
    
    result_s = f"{model_name}: \nACC:{accuracy:.3f} \nSP:{specificity:.3f} \nSS:{sensitivity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f} \nTime:{mean_time:.3f}"
    
    if (verbose):
        print(result_s)
  
    if (save_results):
        ResultsToFile(model_name, result_s)
    
    createConfusionMatrix(cm, model_name, save=save_results)
  
    return accuracy, specificity, sensitivity, precision, f1score

def leaveOneOut2(X, Y, model_name, model_function, model_parameters, epochs=50, batch_size=32, verbose=1, save_results=True):
  loo = LeaveOneOut()
  loo.get_n_splits(X)
  y_predict = []
  times = []
  
  labels = Y
  y = to_categorical(Y)

  contador = 0
  for t_v_i, test_i in loo.split(X):
      if (verbose):
          print(f"LOOP {contador}:")
     
      

      X_train = X[t_v_i]
      y_train = y[t_v_i]

      X_test = X[test_i]
      y_test = y[test_i]
      
      start_time = time.time()
      
      model = model_function(**model_parameters)
      
      if (verbose>=2):
          print(f"{model_name}:")
          print(model.summary())
      
      h = model.fit(X_train, y_train, epochs=epochs ,batch_size=batch_size, verbose=0)
        
      end_time = time.time()
      seconds = end_time - start_time
      times.append(seconds)
      
      if (verbose):
          print("Train Loss ",round(statistics.mean(h.history['loss']),3))
          print("Train Accuracy ",round(statistics.mean(h.history['accuracy']),3))

      y_pre = model(X_test)
      y_pre = [np.argmax(values) for values in y_pre]
      # print(f"Y_true: {np.argmax(y_test, axis=1)} Y_pre: {y_pre}")
      y_predict.append(y_pre)
      
      if (verbose):
          print(f"Eval Accuracy: {CalculateAccuracy(np.argmax(y_test, axis=1), y_pre)}")
      
      contador +=1

  cm = metrics.confusion_matrix(labels, y_predict)
  accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
  mean_time = sum(times) / len(times)
  
  result_s = f"{model_name}: \nACC:{accuracy:.3f} \nSP:{specificity:.3f} \nSS:{sensitivity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f} \nTime:{mean_time:.3f}"
  
  if (verbose):
      print(result_s)

  if (save_results):
      ResultsToFile(model_name, result_s)
  
  createConfusionMatrix(cm, model_name, save=save_results)

  return accuracy, specificity, sensitivity, precision, f1score



def main():
    print("MAIN")
    
    #Set seed for reproducible results
    seed(1)
    
    X, Y = ReadData(light='WL')
    # print(X.shape)
    # print(Y.shape)
    
    # print(Y)
    # for x in X[0:5]:
    #     ShowImage(x)
    
    originalNN_parameters = {'input_size_net':(224,224,3), 'output_size':2}
    leaveOneOut(X, Y, "OriginalNN", createModelMnist, originalNN_parameters)
    
    # data_sets_names = ["X_WL"]
    # data_sets = [X]
    
    # OriginalNN(data_sets_names, data_sets, Y)
    N_training = 50
    X_train = X[:N_training]
    Y_train = Y[:N_training]
    X_test = X[N_training:]
    Y_test = Y[N_training:]
    # autokerasModel(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
  main()