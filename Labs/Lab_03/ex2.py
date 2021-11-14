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
from tensorflow import keras


def load_data():
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


class SignalGenerator:
    """Python class called SignalGenerator to implement the data 
    preparation and preprocessing steps."""
    def __init__(self, 
                keywords, 
                sampling_rate,
                frame_length,
                frame_step,
                num_mel_bins,
                low_freq,
                up_freq,
                num_coefficients,
                bool_mfcc):
        self.keywords = keywords
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.low_freq = low_freq
        self.up_freq = up_freq
        self.num_coefficients = num_coefficients
        self.bool_mfcc = bool_mfcc


    def get_raw_dataset(self, path):

        labels = []
        path_files = []

        for keyword in self.keywords:
            for audio_file in os.listdir(path + "/" + keyword):
                path_files.append(path + "/" + keyword + "/" + audio_file)
                labels.append(keyword)

        path_file_ds = tf.data.Dataset.from_tensor_slices(path_files)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        return tf.data.Dataset.zip((path_file_ds, label_ds))


    def load_and_preprocess_audio(self, audio_path, label):

        # get the tf audio for each recording
        audio = tf.io.read_file(audio_path)
        tf_audio, rate = tf.audio.decode_wav(audio)
        tf_audio = tf.squeeze(tf_audio, 1)

        # to do : pad the audio to 1s

        frame_length = int(self.frame_length * 1e-3 * 16000)
        frame_step = int(self.frame_step * 1e-3 * 16000)

        # compute the stft
        stft = tf.signal.stft(tf_audio, frame_length, frame_step, fft_length=frame_length)

        # extract the spectrogram
        spectrogram = tf.abs(stft)

        if self.bool_mfcc:
            # compute the MFCC
            num_spectrogram_bins = spectrogram.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                            self.num_mel_bins, 
                                                            num_spectrogram_bins, 
                                                            self.sampling_rate, # sampling rate 
                                                            self.low_freq, 
                                                            self.up_freq
                                                    )

            mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :self.num_coefficients]

            return mfccs, label 
        else:
            return stft, label


    def make_dataset(self,path):

        ds = self.get_raw_dataset(path)
        ds = ds.shuffle(200)
        ds = ds.map(self.load_and_preprocess_audio)

        return ds


################################
########### MAIN
################################

signalgenerator = SignalGenerator(
               keywords = ['down','go','left','no','right','stop','up','yes'], 
               sampling_rate = 16000,
               frame_length = 16,
               frame_step = 8,
               num_mel_bins = None,
               low_freq = None,
               up_freq = None,
               num_coefficients = None,
               bool_mfcc = False)

ds = signalgenerator.make_dataset('/content/data/mini_speech_commands')