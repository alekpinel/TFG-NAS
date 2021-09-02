import time

import torch
import torch.nn as nn

from micro import MicroNetwork
from utilsenas import accuracy_binary, reward_accuracy_binary
from nni.retiarii.oneshot.pytorch.enas import EnasTrainer
from torch.utils.data import DataLoader

import numpy as np
from torch.utils.data import Dataset
from shutil import copyfile
import os
import json

from nni.nas.pytorch.fixed import apply_fixed_architecture


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
    
def loadENASMoelXY(X, Y, n_classes=1, num_layers=3, num_nodes=5, dropout_rate=0.1, path='saves/ENAS/model_json'):
    return loadENASModel(n_classes, num_layers, num_nodes, dropout_rate, path)

def loadENASModel(n_classes=1, num_layers=3, num_nodes=5, dropout_rate=0.1, path='saves/ENAS/model_json'):
    model = model = MicroNetwork(num_classes=n_classes, num_layers=num_layers, out_channels=20, num_nodes=num_nodes, dropout_rate=dropout_rate, use_aux_heads=False)
    apply_fixed_architecture(model, path)
    return model

def enasModelFromNumpy(X, Y, epochs=10, n_classes=1, num_layers=3, num_nodes=5, dropout_rate=0.1, saveLoad=True):
    X = np.moveaxis(X, -1, 1)
    Y = np.reshape(Y, (len(Y), 1))
    database = NumpyDataset(X, Y)
    best_model, extra_info = enasModel(database, n_classes=n_classes, epochs=epochs, num_layers=num_layers, saveLoad=saveLoad, num_nodes=num_nodes, dropout_rate=dropout_rate)
    return best_model, extra_info
    
    

def enasModel(database, validation_split=0.3, n_classes=10, epochs=10, num_layers=3, num_nodes=5, dropout_rate=0.1, saveLoad=True):
    print(f"N: {len(database)} X: {database[0][0].shape}")
    
    mutator = None
    ctrl_kwargs = {}
    model = MicroNetwork(num_classes=n_classes, num_layers=num_layers, out_channels=20, num_nodes=num_nodes, dropout_rate=dropout_rate, use_aux_heads=False)
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
    total_epochs = 0
    total_time = 0
    if (saveLoad):
        savepath='saves/ENAS/checkpoint.pt'
        copy_savepath='saves/ENAS/copy_checkpoint.pt'
        if (os.path.isfile(savepath)):
            copyfile(savepath, copy_savepath)
            
            checkpoint = torch.load(savepath)
            trainer.model.load_state_dict(checkpoint['model'])
            trainer.controller.load_state_dict(checkpoint['controller'])
            total_epochs = checkpoint['total_epochs']
            total_time = checkpoint['total_time']
            print(f"Previous epochs: {total_epochs}")
    
    start_time = time.time()
    
    trainer.fit()
    
    end_time = time.time()
    seconds = end_time - start_time
    
    total_time += seconds 
    
    print(f"Total time: {total_time}s")
    
    if (saveLoad):
        trainer.model.to('cpu')
        trainer.controller.to('cpu')
        total_epochs += epochs
        print(f"Total epochs: {total_epochs}")
        state = {
                'model': trainer.model.state_dict(),
                'controller': trainer.controller.state_dict(),
                'total_epochs': total_epochs,
                'total_time': total_time
        }
        
        torch.save(state,savepath)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer.model.to(device)
        trainer.controller.to(device)
    
    trainer.controller.cpu()
    model = trainer.export()
    print(model)
    
    with open('saves/ENAS/model_json', 'w') as json_file:
        json.dump(model, json_file)
    
    info = f"Time: {total_time}\nEpochs: {total_epochs}\n{model}"
    best_model = loadENASModel(n_classes, num_layers, num_nodes, dropout_rate)
    
    return best_model, info

    
    
def numpyToTorch(X, Y):
    return NumpyDataset(X, Y)
    
def torchToNumpy(database):
    train_loader = DataLoader(database, batch_size=len(database))
    
    # train_dataset_array = next(iter(train_loader))[0].numpy()
    train_dataset_array = database.data
    train_dataset_labels = database.targets
    train_dataset_labels = np.array(train_dataset_labels, np.int64)
    
    return (train_dataset_array, train_dataset_labels)
