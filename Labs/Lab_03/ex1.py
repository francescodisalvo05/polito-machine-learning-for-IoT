import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--labels', type=int, required=True, help='model output')
    args = parser.parse_args()

    # initialize seed
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # get_data(args)
    


if __name__ == '__main__':
    main()