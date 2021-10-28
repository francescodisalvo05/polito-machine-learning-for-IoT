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
	spectrogram = tf.io.parse_tensor(byte_spectrogram, out_type=tf.float32)

	# compute the Mel spectrogram with 40mel bins, 20Hz as lower freq and 4kHz as upper freq
	print('--- Extracting the spectrogram from the stft ---')
	time_mfcc = time.time()

	num_spectrogram_bins = spectrogram.shape[-1]
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
						num_spectrogram_bins
						args.num_spectrogram_bins,
						args.sampling_rate,
						args.lower_frequency,
						args.upper_frequency
					)
	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix,1)
	mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

def main():

	parser = ArgumentParser()

	parser.add_argument('--base_path', type=str, default='audio/')
	parser.add_argument('--filename', type=str, default='sftf_res_yes_01.wav')

	parser.add_argument('--num_spectrogram_bins', type=int, default=40)
	parser.add_argument('--sampling_rate', type=int, default=16000)
	parser.add_argument('--lower_frequency', type=int, default=20)
	parser.add_argument('--upper_frequency', type=int, default=4000)

	args = parser.parse_args()

	get_mfcc(args)

if __name__ == '__main__':

	main()
