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


