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
  tp, fp, fn, tn = cmf.ravel()
  
  accuracy = specificity = sensitivity = precision = f1 = 0
  
  if (sum([tp,fp,fn,tn])>0):
      accuracy = (tp + tn) / sum([tp,fp,fn,tn])

  if ((tn+fp)>0):
      specificity = tn / (tn+fp)

  if ((tp+fn)>0):
      sensitivity = tp / (tp+fn)

  if ((tp+fp)>0):
      precision = tp / (tp+fp)
  
  if ((sensitivity+precision)>0):
      f1 = 2 * ((sensitivity*precision)  / (sensitivity+precision))

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
    f = open(resultpath + model_name + ".txt", "w")
    f.write(results)
    f.close()
    
def SummaryString(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary
    
def createConfusionMatrix(cm,name_clf, tipo_de_clas=0, save=True):
    if(tipo_de_clas == 0):
      labels = ["hyperplasic","adenoma/serrated"]
    if(tipo_de_clas == 1):
      labels = ["hyperplasic","serrated","adenoma"]

    con_mat_df = pd.DataFrame(cm,
                      index = labels, 
                      columns = labels)

    figure = plt.figure(figsize=(5, 5))
    sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: "+name_clf)
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
    
def ClearWeights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            ClearWeights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            if var is not None:
                var.assign(initializer(var.shape, var.dtype))
            #use the initializer    
    return model