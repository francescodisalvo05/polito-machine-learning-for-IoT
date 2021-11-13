import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras


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



def get_model(args):

    if args.model == "MLP":
        model = keras.Sequential([
          keras.layers.Flatten(input_shape=(6, 2)),
          keras.layers.Dense(128, activation='relu', name='first_dense'),
          keras.layers.Dense(128, activation='relu', name='second_dense'),
          keras.layers.Dense(1, activation='softmax', name='third_dense')
        ])

    return model


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


    


if __name__ == '__main__':
    main()