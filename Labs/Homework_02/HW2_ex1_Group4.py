import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib

tf.random.set_seed(42)
np.random.seed(42)


class WindowGenerator:
    def __init__(self, input_width, label_step, label_options, mean, std, verbose=False):
        self.input_width = input_width
        self.label_options = label_options
        self.label_step = label_step
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])
        self.verbose = verbose

    def split_window(self, features):
        inputs = features[:, :self.input_width, :] 
        labels = features[:, self.input_width:, :]
        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.label_step, 2])
        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)
        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)
        return inputs, labels

    def make_dataset(self, data, train):
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+self.label_step,
                sequence_stride=1,
                batch_size=32)

        dataset = dataset.map(map_func=self.preprocess)
        if self.verbose:
            print(f"Dataset element: {dataset.element_spec}")
        dataset = dataset.cache() 
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset

class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_output_mae', **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=[2])
        self.count = self.add_weight('count', initializer='zeros')

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


def get_data(step):

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='.', 
        cache_subdir='data'
    )

    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    
    column_indices = [2, 5]
    columns = df.columns[column_indices]
    
    data = df[columns].values.astype(np.float32)
    n = len(data)
    
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    generator = WindowGenerator(input_width=6, label_step=step, \
                                label_options=2, mean=mean, std=std)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    return train_ds, val_ds, test_ds

def get_model(step, alpha):
    model = keras.Sequential([keras.layers.Flatten(input_shape=(6,2)),
                              keras.layers.Dense(units=128 * alpha, activation='relu'),
                              keras.layers.Dense(units=128 * alpha, activation='relu'),
                              keras.layers.Dense(units=step * 2),
                              keras.layers.Reshape([step, 2])])

    return model

def save_tf(model, version):

    SAVING_TF_PATHDIR = 'models_tf/Group4_1_{}'.format(version)

    if not os.path.isdir('models_tf/'):
        os.mkdir('models_tf/')

    if not os.path.isdir(SAVING_TF_PATHDIR):
        os.mkdir(SAVING_TF_PATHDIR)

    model.save(SAVING_TF_PATHDIR)

    return SAVING_TF_PATHDIR

def save_tflite(TF_PATH, version, optimization=None):

    TFLITE_PATHDIR =  'models_tflite/'
    TFLITE_MODEL_PATH =  TFLITE_PATHDIR + 'Group4_th_{}.tflite'.format(version)

    if not os.path.isdir('models_tflite/'):
        os.mkdir('models_tflite/')

    if not os.path.isdir(TFLITE_PATHDIR):
        os.mkdir(TFLITE_PATHDIR)

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = optimization

    tflite_m = converter.convert()

    # save tflite model
    with open(TFLITE_MODEL_PATH, 'wb') as fp:
        fp.write(tflite_m)

    # compress the tflite model and save it
    TFLITE_PATH_COMPRESSED = TFLITE_MODEL_PATH + ".zlib"
    with open(TFLITE_PATH_COMPRESSED, 'wb') as fp:
        compressed_tflite_model = zlib.compress(tflite_m, level=9)
        fp.write(compressed_tflite_model)

    return os.path.getsize(TFLITE_PATH_COMPRESSED) / 1024, TFLITE_MODEL_PATH

def evaluate_tflite(test_ds, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_ds = test_ds.unbatch().batch(1)

    total_errors = []

    for x, y in test_ds:
        # give the input
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        # predict and get the current ground truth
        curr_predictions = interpreter.get_tensor(output_details[0]['index']).squeeze()
        curr_labels = y.numpy().squeeze()

        # as error, get the average for the given window
        curr_errors = np.mean(np.abs(curr_predictions - curr_labels),axis=0) # average by column


        total_errors.append(curr_errors)

    final_error = np.mean(total_errors, axis=0)

    return final_error
      
def print_statistics(size, error):
    print('\n')
    print(f'Model size = {size:.3f}')
    print(f'MAE (temp) = {error[0]:.3f}')
    print(f'MAE (hum)  = {error[1]:.3f}')

def main(args):
  
    if args.version == 'a':
        epochs = 22
        alpha = 0.07
        step = 3
        final_sparsity = 0.85

    elif args.version == 'b':
        epochs = 30
        alpha = 0.08
        step = 9
        final_sparsity = 0.85

    # get dataset
    train_dataset, val_dataset, test_dataset = get_data(step)

    # get the base model (MLP)
    model = get_model(step, alpha)

    # pruning setup
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                               final_sparsity=final_sparsity,
                                                                               begin_step=len(train_dataset) * 5,
                                                                               end_step=len(train_dataset) * 15)}
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model = prune_low_magnitude(model, **pruning_params)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # build & compile & fit
    input_shape = [32, 6, 2] 
    model.build(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=[MultiOutputMAE()])
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
    
    # strip pruning
    model = tfmot.sparsity.keras.strip_pruning(model)

    # save tensorflow model
    TF_PATHDIR = save_tf(model, args.version)
    
    # PTQ
    optimization = [tf.lite.Optimize.DEFAULT]

    # save tflite model
    final_size, TFLITE_MODEL_PATH = save_tflite(TF_PATHDIR, args.version, optimization=optimization)

    # evaluate tflite model on test dataset
    if args.statistics:
        final_error = evaluate_tflite(test_dataset, TFLITE_MODEL_PATH)
        print_statistics(final_size, final_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-v','--version',type=str, choices=['a','b','c'], required=True)
    parser.add_argument('-s','--statistics', type=bool, default=True)
    
    args = parser.parse_args()

    main(args)