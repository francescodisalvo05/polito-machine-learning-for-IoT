import tensorflow as tf

import time

from scipy.io import wavfile
from scipy import signal

import numpy as np

from argparse import ArgumentParser

import os


def get_mfcc(args):

	# load the spectrogram
	print('--- Loading the spectrogram ---')
	byte_spectrogram = tf.io.read_file(args.base_path + args.filename)
	# transform the serialized tensor into a tensor
	print('--- Transforming the serialized tensor into a tensor ---')
	tensor_spectrogram = tf.io.parse_tensor(byte_spectrogram, out_type=tf.float32)

	# compute the Mel spectrogram with 40mel bins, 20Hz as lower freq and 4kHz as upper freq
	print('--- Extracting the spectrogram from the stft ---')
	time_mfcc = time.time()

	num_spectrogram_bins = spectrogram.shape[-1]
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
						args.num_mel_bins,
						args.num_spectrogram_bins,
						args.sampling_rate,
						args.lower_frequency,
						args.upper_frequency
					)


def main():

	parser = ArgumentParser()

	parser.add_argument('--base_path', type=str, default='audio/')
	parser.add_argument('--filename', type=str, default='sftf_res_yes_01.wav')



	args = parser.parse_args()

	get_mfcc(args)

if __name__ == '__main__':

	main()
