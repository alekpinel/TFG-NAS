import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

from datasets import get_dataset
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)
from utilsenas import accuracy, reward_accuracy, accuracy_binary, reward_accuracy_binary
from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import numpy as np
from torch.utils.data import Dataset, DataLoader
from shutil import copyfile

class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        # self.targets = torch.LongTensor(targets)
        self.targets = torch.FloatTensor(targets)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def enasModelFromNumpy(X, Y, epochs=10, n_classes=1, num_layers=3, saveLoad=True):
    X = np.moveaxis(X, -1, 1)
    Y = np.reshape(Y, (len(Y), 1))
    database = NumpyDataset(X, Y)
    model = enasModel(database, n_classes=n_classes, epochs=epochs, num_layers=num_layers, saveLoad=saveLoad)
    return model
    
    
    # test_dl = DataLoader(test_small, batch_size=32, shuffle=False)
    # y_true, y_predict = evaluate_model(test_dl, model)
    

def enasModel(database, validation_split=0.3, n_classes=10, epochs=10, num_layers=3, saveLoad=True):
    print(f"N: {len(database)} X: {database[0][0].shape}")
    
    mutator = None
    ctrl_kwargs = {}
    model = MicroNetwork(num_classes=n_classes, num_layers=num_layers, out_channels=20, num_nodes=5, dropout_rate=0.1, use_aux_heads=False)
    batchsize = 128
    num_epochs = epochs
    log_frequency = 1
    ctrl_kwargs = {"tanh_constant": 1.1}
    
    if (n_classes > 1):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
        
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    trainer = EnasTrainer(model,
                              loss=criterion,
                              metrics=accuracy_binary,
                              reward_function=reward_accuracy_binary,
                              optimizer=optimizer,
                              batch_size=batchsize,
                              num_epochs=num_epochs,
                              dataset=database,
                              log_frequency=log_frequency,
                              ctrl_kwargs=ctrl_kwargs)
    
    if (saveLoad):
        savepath='saves/checkpoint.pt'
        copy_savepath='saves/copy_checkpoint.pt'
        copyfile(savepath, copy_savepath)
        
        checkpoint = torch.load(savepath)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.controller.load_state_dict(checkpoint['controller'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        total_epochs = checkpoint['total_epochs']
        # total_epochs = 0
        print(f"Previous epochs: {total_epochs}")
    
    trainer.fit()
    
    if (saveLoad):
        trainer.model.to('cpu')
        trainer.controller.to('cpu')
        total_epochs += epochs
        print(f"Total epochs: {total_epochs}")
        state = {
                'model': trainer.model.state_dict(),
                'controller': trainer.controller.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                'total_epochs': total_epochs
        }
        
        torch.save(state,savepath)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer.model.to(device)
        trainer.controller.to(device)
    
    # torch.save(trainer, "saves/trainer.pt")
    return trainer.model

    # model = MLP(1)
    # return model
    
    
def numpyToTorch(X, Y):
    return NumpyDataset(X, Y)
    
def torchToNumpy(database):
    train_loader = DataLoader(database, batch_size=len(database))
    
    # train_dataset_array = next(iter(train_loader))[0].numpy()
    train_dataset_array = database.data
    train_dataset_labels = database.targets
    train_dataset_labels = np.array(train_dataset_labels, np.int64)
    
    return (train_dataset_array, train_dataset_labels)

def getData():
    dataset_train, dataset_valid = get_dataset("cifar10")
    
    # print(dataset_train[0])
    
    # dataset_train = torchToNumpy(dataset_train)
    # dataset_valid = torchToNumpy(dataset_valid)
    
    return dataset_train, dataset_valid  

def CalculateAccuracy(Y, Pred):
    booleans = np.equal(Y, Pred)
    n = np.sum(booleans)
    return n/Y.shape[0]


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
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

def main():
    print("MAIN")
    np.random.seed(1)
    
    predictions = []
    yhat = np.array([[0.1], [0.5], [0.9], [0.4]])
    # round to class values
    print('yhat')
    print(yhat)
    yhat = yhat.round()
    # store
    predictions.append(yhat)
    
    predictions = np.vstack(predictions)
    print('predictions')
    print(predictions)
    print(predictions.shape)
    # predictions = np.argmax(predictions, axis=1)
    predictions = np.reshape(predictions, (len(predictions),))
    print('predictions')
    print(predictions)
    print(predictions.shape)
    return  predictions

    
    
    dataset_train, dataset_valid = getData()
    
    train_small, _ = torch.utils.data.random_split(dataset_train, [1000, len(dataset_train)-1000])
    test_small, _ = torch.utils.data.random_split(dataset_valid, [100, len(dataset_valid)-100])
    
    
    model = enasModel(train_small)
    
    
    test_dl = DataLoader(test_small, batch_size=32, shuffle=False)
    y_true, y_predict = evaluate_model(test_dl, model)
    
    
    # print(y_true)
    print(y_predict)
    y_predict = np.argmax(y_predict, axis=1)
    print(y_predict)
    
    print(np.unique(y_predict))
    
    accuracy = CalculateAccuracy(y_true, y_predict)
    print(f"Accuracy: {accuracy}")
    
    return

    X_test = dataset_valid[0][:100]
    Y_test = dataset_valid[1][:100]
    
    # X_test = torch.Tensor(X_test)
    
    # dataset_test = numpyToTorch(X_test, Y_test)
    
    print(f"X_test: {X_test.shape} Y_test: {Y_test.shape}")
    
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    
    # y_true, y_predict = pytorch_predict(model, test_loader)
    
    X_test = torch.from_numpy(X_test)
    print(X_test.size())
    y_predict = model(X_test)
    
    
    print(y_predict)
    print(y_predict.shape)
    
    # y_predict = [1 if val > 0.5 else 0 for val in y_predict]
    y_predict = np.argmax(y_predict, axis=1)
    print(y_predict)
    
    print(np.unique(y_predict))
    
    accuracy = CalculateAccuracy(Y_test, y_predict)
    print(f"Accuracy: {accuracy}")
    
    # cm = metrics.confusion_matrix(Y_test, y_predict)
    # accuracy, specificity, sensitivity, precision, f1score = extraerSP_SS(cm)
    # cm = metrics.confusion_matrix(Y_test, y_predict)
    
    # con_mat_df = pd.DataFrame(cm)

    # figure = plt.figure(figsize=(5, 5))
    # sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix: ")
    # plt.tight_layout()
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    
    # plt.show()
    # plt.close()
    
if __name__ == '__main__':
  main()