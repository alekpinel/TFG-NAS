# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:03:33 2021

@author: alekp
"""


from tensorflow import keras
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics import plot_confusion_matrix, balanced_accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from keras.utils import plot_model
import tensorflow as tf

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torchinfo import summary
import torch.nn as nn

import logging
import os


mainpath = "./" #Local
#mainpath = "/content/drive/My Drive/Colab Notebooks/TFG/" #Colab


def ReadData(light='WL', input_size = (224,224)):
    imagepath = mainpath + "data/" + light + '/'
    
    paths = [imagepath + 'adenoma', imagepath + 'hyperplasic', imagepath + 'serrated']
    
    labels = range(len(paths))

    valid_images = [".jpg"]
    X = []
    X_names = []
    Y = []
    
    for p, y in zip(paths, labels):
        lst = os.listdir(p)
        lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        for f in lst:
            ext = os.path.splitext(f)[1]
    
            if ext.lower() not in valid_images:
                continue
                
            img = load_img(os.path.join(p,f),target_size=input_size)
            X_names.append(f.split(".")[0])
            img_array = img_to_array(img)
            X.append(img_array)
            Y.append(y)
    
    
    X = np.asarray(X)
    X /= 255
    Y = np.array(Y)
    
    X, Y = ShuffleData(X, Y)
    # Y = convertToBinary(Y)
    
    return X, Y

# Shuffle the data and the labels
def ShuffleData(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]
    return X, Y


def convert_Y(y_labels):
    encoder = LabelEncoder()
    encoder.fit(y_labels)
    encoded_Y = encoder.transform(y_labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    converted_y = to_categorical(encoded_Y)
    #print(converted_y.shape)
    return converted_y

#Instead of three classes, we will only use 2
def convertToBinary(y):
    y2 = np.array(y)
    y2[(y==0)] = 0
    y2[(y==2)] = 0
    y2[(y==1)] = 1
    return y2


def CalculateAccuracy(Y, Pred):
    booleans = np.equal(Y, Pred)
    n = np.sum(booleans)
    return n/Y.shape[0]

def extraerSP_SS(cmf):
  # tp, fp, fn, tn = cmf.ravel()
  tp, fn, fp, tn = cmf.ravel()
  
  # print(f"tp {tp} fp {fp} fn {fn} tn {tn}")
  
  return calculate_metrics(tp, fn, fp, tn)

def calculate_metrics(tp, fn, fp, tn):
  
  accuracy = specificity = sensitivity = precision = f1 = 0
  
  if (sum([tp,fp,fn,tn])>0):
      accuracy = (tp + tn) / sum([tp,fp,fn,tn])
      
  if ((tp+fn)>0):
      sensitivity = tp / (tp+fn)

  if ((tn+fp)>0):
      specificity = tn / (tn+fp)

  if ((tp+fp)>0):
      precision = tp / (tp+fp)
  
  f1 = 0.35*sensitivity + 0.25*accuracy + 0.2*specificity + 0.2*precision

  return round(accuracy,3), round(specificity,3),round(sensitivity,3), round(precision,3), round(f1, 3)

    
#Shows an Image using Matplotlib 
def ShowImage(img, title=None):
    plt.imshow(img, cmap='gray')
    if (title != None):
        plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()
    
#Show multiple images
def ShowImages(images, titles=None):
    nimages = len(images)
    
    for i in range(nimages):
        plt.subplot(1, nimages, i+1)
        plt.imshow(images[i],cmap='gray')
        if (titles is not None):
            plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        
    plt.tight_layout()
    plt.show()

def ResultsToFile(model_name, results):
    resultpath = mainpath + "results/"
    f = open(resultpath + model_name + ".txt", "w", encoding="utf-8")
    f.write(results)
    f.close()
    
def SummaryString(model, api='tensorflow'):
    if (api == 'tensorflow'):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary
    else:
        model_stats =  summary(model, input_size=(32, 3, 224, 224), verbose=0, depth =10)
        # model_str = f"{str(model)}\n{str(model_stats)}"
        model_str = f"{str(model_stats)}"
        return model_str
    
def createConfusionMatrix(cm,name_clf, tipo_de_clas=0, save=True):
    if(tipo_de_clas == 0):
      labels = ["adenoma/serrated", "hyperplasic"]
    if(tipo_de_clas == 1):
      labels = ["hyperplasic","serrated","adenoma"]

    con_mat_df = pd.DataFrame(cm,
                      index = labels, 
                      columns = labels)

    figure = plt.figure(figsize=(5, 5))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix: ")
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    if (save):
        resultpath = mainpath + "results/"
        plt.savefig(resultpath + name_clf + "_cm.png",  bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def PlotModelToFile(model, model_name):
    plotpath = mainpath + "results/" + model_name + "_plot.png"
    plot_model(model, plotpath)
    
def ClearWeightsTensorflow(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            ClearWeightsTensorflow(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it
                 
            if not hasattr(init_container, 'kernel'):
                continue

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            if var is not None:
                var.assign(initializer(var.shape, var.dtype))
            #use the initializer    
    return model

def ClearWeightsPytorch(model):
    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
    
    model.apply(weight_reset)
    return model

class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.FloatTensor(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

# evaluate the model
def predict_pytorch(X, model):
    Y = np.zeros((len(X)))
    X = np.moveaxis(X, -1, 1)
    database = NumpyDataset(X, Y)
    test_dl = DataLoader(database, batch_size=32, shuffle=False)
    
    predictions = list()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i, (inputs, targets) in enumerate(test_dl):
        device_inputs, device_targets = inputs.to(device), targets.to(device)
        
        # evaluate the model on the test set
        with torch.no_grad():
            yhat = model(device_inputs)
        # retrieve numpy array
        yhat = yhat.detach().cpu().numpy()
        # round to class values
        # print('yhat')
        # print(yhat)
        yhat = yhat.round()
        # store
        predictions.append(yhat)
    
    predictions = np.vstack(predictions)
    # print('predictions')
    # print(predictions)
    # print(predictions.shape)
    # predictions = np.argmax(predictions, axis=1)
    predictions = np.reshape(predictions, (len(predictions),))
    # print('predictions')
    # print(predictions)
    # print(predictions.shape)
    return  predictions

def train_model_pytorch(model, X, Y, epochs=50, batch_size=32):
    original_X = X
    original_Y = Y
    
    X = np.moveaxis(X, -1, 1)
    Y = np.reshape(Y, (len(Y), 1))
    database = NumpyDataset(X, Y)
    train_dl = DataLoader(database, batch_size=batch_size, shuffle=True)
    
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()
    # enumerate epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            device_inputs, device_targets = inputs.to(device), targets.to(device)
            
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(device_inputs)
            # calculate loss
            loss = criterion(yhat, device_targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
            
        predictions = predict_pytorch(original_X, model)
        predictions = [1 if val > 0.5 else 0 for val in predictions]
        accuracy = CalculateAccuracy(original_Y, predictions)
        print(f"Epoch {epoch}/{epochs} loss: {running_loss/n_batches} accuracy: {accuracy}")

# evaluate the model
def evaluate_model_pytorch(test_dl, model):
    predictions, actuals = list(), list()
    with torch.no_grad():
        model.eval()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            yhat = yhat.round()
            # store
            predictions.append(yhat)
            actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # # calculate accuracy
    # acc = np.accuracy_score(actuals, predictions)
    return actuals, predictions

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)