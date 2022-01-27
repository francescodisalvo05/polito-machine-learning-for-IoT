"""
The Mini Speech Command dataset collects 8000 samples of eight keywords 
('down', 'no', 'go', 'yes', 'stop', 'up', 'right', 'left'), 1000 samples 
per label. Each sample is recorded at 16kHz with an Int16 resolution and 
has a variable duration (1s the longest).

Write a Python script to train and evaluate different models for keyword 
spotting on the Mini Speech Command dataset.
"""

import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import zlib
from scipy import signal


RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False, resampling_rate = None):

        self.labels = labels

        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = (frame_length) // 2 + 1
        rate = self.resampling_rate if self.resampling_rate else self.sampling_rate
           
        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def apply_resampling(self, audio):
        audio = signal.resample_poly(audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate:
            audio = tf.numpy_function(self.apply_resampling, [audio], tf.float32)

        return audio, label_id


    def pad(self, audio):
        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate
        zero_padding = tf.zeros([rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


def load_data(options, resampling):

    mini_speech_command_zip = tf.keras.utils.get_file(
                        origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                        fname = "mini_speech_commands.zip",
                        extract=True,
                        cache_dir='.',
                        cache_subdir='data'
                    )
    # it will download a folder and its subfolders
    # data > mini_speech_commands > [down/, go/, left/, no/, right/, stop/, us/, yes/]
    csv_path, _ = os.path.splitext(mini_speech_command_zip)

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    with open("./kws_train_split.txt", "r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  # len 6400
    with open("./kws_val_split.txt", "r") as fp:
        val_files = [line.rstrip() for line in fp.readlines()]  # len 800
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800

    labels = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

    generator = SignalGenerator(labels, sampling_rate=16000, resampling_rate=resampling, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds, test_ds

def get_dscnn(alpha=1):

    strides = [2,1] 
    units = 8
    
    model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=units)
              ])

    return model

def save_tf(model, version):

    SAVING_TF_PATHDIR = 'models_tf/Group4_kws_{}'.format(version)

    if not os.path.isdir('models_tf/'):
        os.mkdir('models_tf/')

    if not os.path.isdir(SAVING_TF_PATHDIR):
        os.mkdir(SAVING_TF_PATHDIR)

    model.save(SAVING_TF_PATHDIR)

    return SAVING_TF_PATHDIR
    
def save_tflite(TF_PATH, version, optimization=None):

    TFLITE_PATHDIR =  'models_tflite/'
    TFLITE_MODEL_PATH =  TFLITE_PATHDIR + 'Group4_kws_{}.tflite'.format(version)

    if not os.path.isdir('models_tflite/'):
        os.mkdir('models_tflite/')

    if not os.path.isdir(TFLITE_PATHDIR):
        os.mkdir(TFLITE_PATHDIR)

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.experimental_enable_resource_variables = True

    if optimization is not None:
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

def evaluate_tflite(test_ds, TFLITE_MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_ds = test_ds.unbatch().batch(1)

    total_count, correct_count = 0, 0

    for x, y in test_ds:
        # give the input
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        # predict and get the current ground truth
        curr_prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
        curr_label = y.numpy().squeeze()

        curr_prediction = np.argmax(curr_prediction_logits)

        if curr_prediction == curr_label:
          correct_count += 1
        
        total_count += 1

    return correct_count / total_count
          
def print_statistics(size, error):
    print('\n')
    print(f'Model size = {size:.3f} kB')
    print(f'Accuracy = {error * 100:.3f} %')

def main(args):

    if args.version == 'a':
        epochs = 25
        alpha = 0.85
        input_shape = [32,49,10,1]
        resampling = None

        MFCC_OPTIONS = {
            'frame_length': 640, 'frame_step': 320, 'mfcc': True, 'lower_frequency': 20, 
            'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
        }

        learning_rate = 0.005


    elif args.version in ['b','c']: 
        epochs = 25
        alpha = 0.3
        input_shape = [32,65,10,1]
        resampling = 8000

        MFCC_OPTIONS = {
            'frame_length': 240, 'frame_step': 120, 'mfcc': True, 'lower_frequency': 20, 
            'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
        }

        learning_rate = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-2,
                        decay_steps=10000,
                        decay_rate=0.9
                      )
                      
        
    train_ds, val_ds, test_ds = load_data(MFCC_OPTIONS, resampling)

    model = get_dscnn(alpha)

    model.build(input_shape=input_shape)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=keras.metrics.SparseCategoricalAccuracy())

    model.fit(train_ds, epochs=epochs, validation_data=val_ds)   

    TF_PATHDIR = save_tf(model, args.version)

    # PTQ Optimization
    optimization = [tf.lite.Optimize.DEFAULT]
    
    final_size, TFLITE_MODEL_PATH = save_tflite(TF_PATHDIR, args.version, optimization=optimization)

    if args.statistics:
        final_error = evaluate_tflite(test_ds, TFLITE_MODEL_PATH)
        print_statistics(final_size, final_error)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-v','--version',type=str, choices=['a','b','c'], required=True)
    parser.add_argument('-s','--statistics', type=bool, default=True)
    
    args = parser.parse_args()

    main(args)
