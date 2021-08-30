import os



import tensorflow as tf
from auto_cnn.gan import AutoCNN

import numpy as np
import keras

import random

from sklearn.model_selection import train_test_split
from auto_cnn.cnn_structure import SkipLayer, PoolingLayer, CNN, Layer, get_layer_from_string


def get_binary_output_function():

    def output_function(inputs):
        out = tf.keras.layers.Flatten()(inputs)

        return tf.keras.layers.Dense(1, activation='sigmoid')(out)

    return output_function

def test_cnn_architecture(X, Y,  architecture_string, dir_name='test', epochs=50):
    
    batch_size=1
    layers = get_layer_from_string(architecture_string)
    optimizer = tf.keras.optimizers.Adam()
    loss = keras.losses.binary_crossentropy
    metrics = ('accuracy',)
    extra_callbacks = None
    main_dir = f'./auto_cnn/outputs/{dir_name}/'
    logs_dir = f'{main_dir}logs/train_data'
    checkpoint_dir = f'{main_dir}checkpoints'
    
    cnn = CNN(X.shape[1:], get_binary_output_function(), layers, optimizer=optimizer, loss=loss,
                   metrics=metrics, extra_callbacks=extra_callbacks, logs_dir=logs_dir,
                   checkpoint_dir=checkpoint_dir)
    model = cnn.get_trained_model(X, Y, batch_size, epochs)
    return model
    
    
def auto_cnn_test(X, Y, dir_name='test', val_percent=0.3, epochs=1, population_size=5, maximal_generation_number=1, crossover_probability = .9, mutation_probability = .2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=val_percent, stratify=Y)
    print(f"X_train: {X_train.shape} Y_train: {Y_train.shape}")
    print(f"X_val: {X_train.shape} Y_val: {Y_train.shape}")
    
    values = X_train.shape[0]

    data = {'x_train': X_train[:values], 'y_train': Y_train[:values], 'x_test': X_test, 'y_test': Y_test}

    batch_size=1
    main_dir = f'./auto_cnn/outputs/{dir_name}/'
    population_size=population_size
    maximal_generation_number=maximal_generation_number
    output_layer=get_binary_output_function()
    epoch_number= epochs
    optimizer = tf.keras.optimizers.Adam()
    loss = keras.losses.binary_crossentropy
    metrics = ('accuracy',)
    crossover_probability = crossover_probability
    mutation_probability = mutation_probability
    mutation_operation_distribution = None
    fitness_cache = f'{main_dir}fitness.json'
    extra_callbacks = None
    logs_dir = f'{main_dir}logs/train_data'
    checkpoint_dir = f'{main_dir}checkpoints'
    population_file = f'{main_dir}population.json'

    a = AutoCNN(population_size=population_size,maximal_generation_number=maximal_generation_number,
                dataset=data, output_layer=output_layer, epoch_number=epoch_number, optimizer=optimizer, loss=loss,
                metrics=metrics, crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                mutation_operation_distribution=mutation_operation_distribution, fitness_cache=fitness_cache, extra_callbacks=extra_callbacks,
                logs_dir=logs_dir, checkpoint_dir=checkpoint_dir,batch_size=batch_size, population_file=population_file)
    best_model = a.run()
    best_model = best_model.get_trained_model(X, Y, batch_size, epochs)
    return best_model
    
    
def data_to_binary(datax, datay):
    y = []
    x = []	
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
        if datay[i] == 0 or datay[i] == 1:
            if datay[i] == 0:
                y.append(0)
            else:
                y.append(1)
            # x.append(np.array([1, datax[i][0], datax[i][1]]))
            x.append(datax[i])
			
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    return x,y

def main():
    tf.get_logger().setLevel('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    random.seed(42)
    tf.random.set_seed(42)
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(x_train.shape)
    
    x_train, y_train = data_to_binary(x_train, y_train)
    x_test, y_test = data_to_binary(x_test, y_test)
    
    print(x_train.shape)
    print(np.unique(y_train))
    
    auto_cnn_test(x_train, y_train)

if __name__ == '__main__':
    main()
