"""
The Jena Climate dataset is a weather time series dataset containing 14 features (e.g. temperature, 
atmospheric pressure, humidity) collected every 10 minutes from 2009 to 2016.

Write a Python script to train and evaluate different models for temperature forecasting on the 
Jena Climate dataset. Select as input features of your model six consecutive temperature and 
humidity measurements. The model should predict the temperature for the next time instant.
"""

import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from prettytable import PrettyTable



# draft provided by the instructor
class WindowGenerator:
    """Python class that implement the data preparation and preprocessing stages. 
    The output is a tf.data.Dataset composed by six-value temperature and humidity windows 
    (the shape of each window is 6×2) and the temperature labels for each window. 
    Each window is normalized with the mean and standard deviation of the training set. 
    """
    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-1, :]

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+1,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


def get_data(args):

    # get dataset
    jena_dataset_zip = tf.keras.utils.get_file(
                            origin = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
                            fname = "jena_climate_2009_2016.csv.zip",
                            extract=True,
                            cache_dir='.',
                            cache_subdir='data'
                        )
    
    csv_path, _ = os.path.splitext(jena_dataset_zip)
    df = pd.read_csv(csv_path)

    # get only the columns #2 and #5 and convert to Float 32
    columns = df.columns[[2, 5]]
    dataset = df[columns].values.astype(np.float32)

    # split the dataset into train/val/test
    N = len(dataset)

    train_idx, val_idx = int(N*0.7), int(N*0.9)
    train_data = dataset[0:train_idx]
    val_data = dataset[train_idx:val_idx]
    test_data = dataset[val_idx:]

    # normalize everything with the values of the
    # training data
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    # fixed
    window_length = 6

    generator = WindowGenerator(window_length, args.labels, mean, std)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    return train_ds, val_ds, test_ds



def get_model(model_name):

    if model_name == "MLP":
        model = keras.Sequential([
          keras.layers.Flatten(),
          keras.layers.Dense(128, activation='relu', name='first_dense'),
          keras.layers.Dense(128, activation='relu', name='second_dense'),
          keras.layers.Dense(1, name='third_dense')
        ])

    elif model_name == "CNN-1D":
      model = keras.Sequential([
          keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
          keras.layers.Flatten(),
          keras.layers.Dense(64, activation='relu'),
          keras.layers.Dense(1)
      ])

    elif model_name == "LSTM":
      model = keras.Sequential([
          tf.keras.layers.LSTM(units=64),
          keras.layers.Flatten(),
          keras.layers.Dense(1)
      ])

    return model


def train_model(model, train_data, val_data, optimizer, loss, metrics, bs, epochs):
    
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    model.fit(train_data, validation_data = val_data, batch_size=bs, epochs=epochs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--labels', type=int, required=True, help='model output')
    args = parser.parse_args()

    # initialize seed
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_ds, val_ds, test_ds = get_data(args)

    # plotting variables
    models = ["MLP", "CNN-1D", "LSTM"]
    mae = []
    num_params = []

    # train all the models for 1 epoch (just to see the pipeline)
    for model_name in models:
        print(f'\n---------- {model_name}')
        model = get_model(model_name)
        history_model = train_model(model=model, train_data=train_ds, val_data=val_ds, optimizer='adam', loss='mse', metrics=['mae'], bs=32, epochs=1)

        # evaluate on the test set
        mae.append(model.evaluate(test_ds, batch_size=32)[1])

        # add final_mae
        mae.append(history_model.history['mae'][-1])
        # add num_params
        num_params.append(model.count_params())

        # save model
        run_model = tf.function(lambda x: model(x))
        concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
        # Signatures define the input and output types for a computation
        model.save(model_name, signatures=concrete_func)

    print('\n ')
    # compare the #params and the test MAE
    mae_params_table = PrettyTable(["Model", "#Params", "MAE"])
    
    for idx, model_name in enumerate(models):
        mae_params_table.add_row([model_name, num_params[idx], mae[idx]])

    print(mae_params_table)

    # with one epoch
    """
    +--------+---------+--------------------+
    | Model  | #Params |     Final MAE      |
    +--------+---------+--------------------+
    |  MLP   |  18305  | 0.8805603384971619 |
    | CNN-1D |  16961  | 3.2827725410461426 |
    |  LSTM  |  17217  | 0.8211777210235596 |
    +--------+---------+--------------------+
    """


if __name__ == '__main__':
    main()