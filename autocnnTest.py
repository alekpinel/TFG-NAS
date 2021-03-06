import tensorflow as tf
from auto_cnn.gan import AutoCNN
import keras
from sklearn.model_selection import train_test_split
from auto_cnn.cnn_structure import SkipLayer, PoolingLayer, CNN, Layer, get_layer_from_string
import os, json
import time


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
    main_dir = f'./saves/AutoCNN/{dir_name}/'
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
    main_dir = f'./saves/AutoCNN/{dir_name}/'
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
    extra_info_path = f'{main_dir}info.json'
    
    total_generations = 0
    total_time = 0
    if os.path.exists(extra_info_path):
            with open(extra_info_path) as json_file:
                info = json.load(json_file)
                total_generations = info[0]
                total_time = info[1]

    
    a = AutoCNN(population_size=population_size,maximal_generation_number=maximal_generation_number,
                dataset=data, output_layer=output_layer, epoch_number=epoch_number, optimizer=optimizer, loss=loss,
                metrics=metrics, crossover_probability=crossover_probability, mutation_probability=mutation_probability,
                mutation_operation_distribution=mutation_operation_distribution, fitness_cache=fitness_cache, extra_callbacks=extra_callbacks,
                logs_dir=logs_dir, checkpoint_dir=checkpoint_dir,batch_size=batch_size, population_file=population_file)
    
    start_time = time.time()
    
    best_model = a.run()
    best_model = best_model.get_trained_model(X, Y, batch_size, epochs)
    
    end_time = time.time()
    seconds = end_time - start_time
    
    total_time += seconds 
    total_generations += maximal_generation_number

    with open(extra_info_path, 'w') as json_file:
        extra_info = [total_generations, total_time]
        json.dump(extra_info, json_file)
    
    extra_info_str = f"Total generations: {total_generations}\nTotal time: {total_time}"
    print(extra_info_str)
    
    return best_model, extra_info_str
