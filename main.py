# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:49:23 2020

@author: alekp
"""

import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import numpy as np

import keras
from sklearn.model_selection import train_test_split, LeaveOneOut
from numpy.random import seed
import time
from tensorflow.keras.utils import to_categorical
import statistics
from sklearn import metrics
from keras.models import clone_model
import logging
import autokeras as ak

from utils import ReadData, CalculateAccuracy, extraerSP_SS, ResultsToFile, createConfusionMatrix, set_tf_loglevel
from utils import convertToBinary, SummaryString, PlotModelToFile, ClearWeightsTensorflow, NumpyDataset, predict_pytorch, train_model_pytorch, ClearWeightsPytorch
from original_nn import OriginalNN 
from autokeras_model import autokerasModel
from autocnnTest import auto_cnn_test, test_cnn_architecture
import torch


import sys
sys.path.insert(0, './enas')
from enasTest import enasModel, enasModelFromNumpy, loadENASMoelXY

def Compile(model):
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

def NASExperiment(X_train, X_test, Y_train, Y_test, model_name, NAS_function, NAS_parameters, epochs=50, batch_size=32, verbose=1, save_results=True, clearModel=True, api='tensorflow'):
    if (verbose):
        print(f"X_train: {X_train.shape} X_test: {X_test.shape}")
    
    
    # Apply NAS to search the architecture
    start_time = time.time()
    
    NAS_parameters['X']=X_train
    NAS_parameters['Y']=Y_train
    
    result = NAS_function(**NAS_parameters)
    
    if (type(result) is tuple):
        NAS_model = result[0]
        extra_info = result[1]
    else:
        NAS_model = result
        extra_info = ""
    
    SaveModel(NAS_model, model_name,api)
    
    
    end_time = time.time()
    seconds = end_time - start_time
    searching_time = seconds
    
    if (verbose):
        print(f"{model_name}:")
        print(SummaryString(NAS_model, api))
        
    result_s = f"{model_name}:"
    result_s += SummaryString(NAS_model, api)
    result_s += f"\n{extra_info}\n"
    
    cm, fit_time = holdOut(X_train, X_test, Y_train, Y_test, NAS_model, clearModel=clearModel, epochs=epochs, batch_size=batch_size, verbose=1, api=api)
    
    # Extract the results and show them
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSS:{sensitivity:.3f} \nSP:{specificity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f} \nSearching time:{searching_time:.3f} \nFitting time:{fit_time:.3f}"
    result_s += statistics_s
    
    if (verbose):
        print(statistics_s)
    
    if (save_results):
        ResultsToFile(model_name, result_s)
    
    createConfusionMatrix(cm, model_name, save=save_results)

# Realizes Leave One Out only in the test set, while the training is always used
def leaveOneOut(X_train, X_test, Y_train, Y_test, original_model, epochs=50, batch_size=32, verbose=1, clear=True, api='tensorflow'): 
    loo = LeaveOneOut()
    loo.get_n_splits(X_test)
    y_predict = []
    times = []
    
    labels_test = Y_test
  
    contador = 0
    for t_v_i, test_i in loo.split(X_test):
        if (verbose):
            print(f"LOOP {contador}:")
        
        # print(f"X_train size: {X_train.shape} X_test[t_v_i] {X_test[t_v_i].shape}")
        X_fold_train = np.concatenate((X_train, X_test[t_v_i]))
        Y_fold_train = np.concatenate((Y_train, Y_train[t_v_i]))
  
        X_fold_test = X_test[test_i]
        Y_fold_test = Y_test[test_i]
        
        new_model = clone_model(original_model)
        # new_model = ClearWeights(original_model)
        Compile(new_model)
        
        start_time = time.time()
        
        h = new_model.fit(X_fold_train, Y_fold_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        end_time = time.time()
        seconds = end_time - start_time
        times.append(seconds)
        
        if (verbose):
            print("Train Loss ",round(statistics.mean(h.history['loss']),3))
            print("Train Accuracy ",round(statistics.mean(h.history['accuracy']),3))
            print(f"Train Time {seconds}")
  
        y_pre = new_model(X_fold_test)
        y_pre = [1 if val > 0.5 else 0 for val in y_pre]
        # print(f"Y_true: {Y_fold_test} Y_pre: {y_pre}")
        y_predict.append(y_pre)
        
        if (verbose):
            print(f"Eval Accuracy: {CalculateAccuracy(Y_fold_test, y_pre)}")
        
        contador +=1
  
    cm = metrics.confusion_matrix(labels_test, y_predict)
    mean_time = sum(times) / len(times)
    
    return cm, mean_time

# Realizes Leave One Out only in the test set, while the training is always used
def holdOut(X_train, X_test, Y_train, Y_test, original_model, epochs=50, batch_size=32, verbose=1, clearModel=True, api='tensorflow'): 
    start_time = time.time()
    
    if (clearModel):
        new_model = ClearModel(original_model, api=api)
        TrainModel(new_model, X_train, Y_train, epochs=epochs, batch_size=batch_size, api=api)
    else:
        new_model = original_model
    
    end_time = time.time()
    seconds = end_time - start_time
    mean_time = seconds
    
    EvaluateModel(new_model, X_train, Y_train, api=api, printResults=True, batch_size=batch_size)
    cm = EvaluateModel(new_model, X_test, Y_test, api=api, printResults=False, batch_size=batch_size)
    
    return cm, mean_time

def SaveModel(model, model_name, api='tensorflow'):
    if (api=='tensorflow'):
        model.save(f'saves/{model_name}.h5')
    elif(api=='pytorch'):
        torch.save(model, f'saves/{model_name}.h5')

def LoadModel(X, Y, model_name, api='tensorflow'):
    if (api=='tensorflow'):
        return keras.models.load_model(f'saves/{model_name}.h5', custom_objects=ak.CUSTOM_OBJECTS)
    elif(api=='pytorch'):
        return torch.load(f'saves/{model_name}.h5')

def ClearModel(model, api='tensorflow'):
    if (api=='tensorflow'):
        model = ClearWeightsTensorflow(model)
        Compile(model)
        return model
    elif(api=='pytorch'):
        model = ClearWeightsPytorch(model)
        return model
        

def TrainModel(model, X_train, Y_train, epochs=50, batch_size=32, api='tensorflow'):
    if (api=='tensorflow'):
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
        model.evaluate(X_train, Y_train, batch_size=batch_size)
        
    elif(api=='pytorch'):
        train_model_pytorch(model, X_train, Y_train, epochs=epochs, batch_size=batch_size)

def EvaluateModel(model, X_test, Y_test, api='tensorflow', printResults=False, batch_size=32):
    if (api=='tensorflow'):
        y_predict = model.predict(X_test, batch_size=batch_size)
    elif(api=='pytorch'):
        y_predict = predict_pytorch(X_test, model)
        
    # print(y_predict)
    y_predict = [1 if val > 0.5 else 0 for val in y_predict]
    # print(y_predict)
    # print(Y_test)
    cm = metrics.confusion_matrix(Y_test, y_predict)
    accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
  
    cm = metrics.confusion_matrix(Y_test, y_predict)
    
    if (printResults):
        # Extract the results and show them
        accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
        statistics_s = f"\nRESULTS: \nACC:{accuracy:.3f} \nSS:{sensitivity:.3f} \nSP:{specificity:.3f} \nPr:{precision:.3f} \nScore:{f1score:.3f}"
        print(statistics_s)
        createConfusionMatrix(cm, "Model", save=False)
    return cm



def main():
    print("MAIN")
    
    #Set seed for reproducible results
    seed(3)
    tf.random.set_seed(3)
    torch.manual_seed(3)
    
    X, Y = ReadData(light='NBI')
    
    test_percent=0.3
    
    # Split the data and prepare the binary
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_percent, stratify=Y)
    Y_train = convertToBinary(Y_train)
    Y_test = convertToBinary(Y_test)
    
    
    ########################### Hand-made neural network  ###########################
    
    originalNN_parameters = {'input_size_net':(224,224,3), 'output_size':1}
    # NASExperiment(X_train, X_test, Y_train, Y_test, "OriginalNN 12", OriginalNN, originalNN_parameters, batch_size=32,epochs=100)
    
    
    ########################### ENAS - Reinforcement Learning  ###########################
    
    enas_parameters = {'epochs':1, 'num_layers':3, 'saveLoad':False, 'num_nodes':2, 'dropout_rate':0.4}
    NASExperiment(X_train, X_test, Y_train, Y_test, "ENAS 3L 2N E1", enasModelFromNumpy, enas_parameters, clearModel=True, api='pytorch', batch_size=16,epochs=100)
    
    
    ########################### Auto-Keras - Bayesian Optimization  ###########################
    
    autokeras_parameters = {'validation_split':0.3, 'epochs':100, 'max_trials':200, 'overwrite':False}
    # NASExperiment(X_train, X_test, Y_train, Y_test, "Autokeras 200", autokerasModel, autokeras_parameters, clearModel=True, api='tensorflow', batch_size=1,epochs=100)
    
    
    ########################### Auto CNN - Evolutive Algorithm  ###########################
    
    auto_cnn_parameters = {'val_percent':0.3, 'epochs':10, 
                'population_size':10, 'maximal_generation_number':100, 
                'crossover_probability':.9, 'mutation_probability':.2, 'dir_name':'tfg-10P-10E'}
    # NASExperiment(X_train, X_test, Y_train, Y_test, "Auto_CNN 10P-10E G100", auto_cnn_test, auto_cnn_parameters, clearModel=True, api='tensorflow', batch_size=32,epochs=100)
    
    
    
    
    archictecture_auto_cnn_parameters = {'architecture_string':'32-128',
                                         'dir_name':'tfg', 'epochs':0}
    # NASExperiment(X_train, X_test, Y_train, Y_test, "Auto_CNN G2", test_cnn_architecture, archictecture_auto_cnn_parameters, clearModel=True, api='tensorflow', batch_size=32,epochs=100)
    
    saved_model_name = "ENAS 3L 2N E20"
    api = 'pytorch'
    saved_model_parameters = {'model_name':saved_model_name, 'api':api}
    # NASExperiment(X_train, X_test, Y_train, Y_test, saved_model_name, LoadModel, saved_model_parameters, clearModel=True, api=api, batch_size=16,epochs=100)
    
    saved_ENAS_parameters = {'num_layers':3, 'num_nodes':2, 'dropout_rate':0.4}
    # NASExperiment(X_train, X_test, Y_train, Y_test, "loaded enas model", loadENASMoelXY, saved_ENAS_parameters, clearModel=True, api='pytorch', batch_size=16,epochs=100)
    
    
    

if __name__ == '__main__':
  main()