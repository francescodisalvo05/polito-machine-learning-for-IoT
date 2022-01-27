import pandas as pd
import numpy as np

import time
from datetime import datetime
import os
from argparse import ArgumentParser

import math

import tensorflow as tf
from subprocess import Popen

from scipy import signal
from scipy.io import wavfile

import io


Popen('sudo sh -c "echo performance >' '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', shell=True).wait()


def get_mfcc(slow, input, frame_length, frame_step, num_mel_bins, low_freq, up_freq, num_coefficients, rate):
    
    # get all the files in the given directory
    files = os.listdir(input)

    frame_length = int(frame_length * 1e-3  * 16000)
    frame_step = int(frame_step * 1e-3 * 16000)

    execution_times = []
    mfccs_list = []

    # calculate it just once in advance
    if not slow:
        # num_spectrogram_bins = floor(frame_length/2) + 1
        num_spectrogram_bins = math.floor(frame_length/2) + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                        num_mel_bins, 
                                                        num_spectrogram_bins, 
                                                        rate, # sampling rate 
                                                        low_freq, 
                                                        up_freq
                                                )
    for file in files:

        # if we do not perform resampling
        if rate == 16000:
             # read it with tensorflow
             audio = tf.io.read_file(input + '/' + file)

        # otherwise we resample
        else:
            _, audio = wavfile.read(input + "/" + file)
            audio = signal.resample_poly(audio, 1, 2)

            # write the resampled audio on a buffer
            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            wavfile.write(byte_io, rate, audio.astype(np.int16))
            audio = byte_io.read()

        tf_audio, rate_audio = tf.audio.decode_wav(audio)
        tf_audio = tf.squeeze(tf_audio, 1)

        # start curr_timer 
        start_timer = time.time()

        # compute the stft
        stft = tf.signal.stft(tf_audio, frame_length, frame_step, fft_length=frame_length)

        # extract the spectrogram
        spectrogram = tf.abs(stft)

        if slow:
            num_spectrogram_bins = spectrogram.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                                        num_mel_bins, 
                                                        num_spectrogram_bins, 
                                                        16000, # default sampling rate 
                                                        low_freq, 
                                                        up_freq
                                                )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :num_coefficients]
        mfccs_list.append(mfccs)
        execution_times.append(time.time() - start_timer)

    average_time = np.mean(execution_times) * 1000.0 # ms

    experiment = 'Slow' if slow else 'Fast'
    print(f'{experiment} = {average_time:.2f} ms')

    return mfccs_list


def get_snr(mfccs_slow_arr,mfccs_fast_arr):
    results = []
    for i in range (len(mfccs_slow_arr)):
        results.append(20*np.log( (np.linalg.norm(mfccs_slow_arr[i])) /((np.linalg.norm(mfccs_slow_arr[i] - mfccs_fast_arr[i] + 10e-6)))))
    return np.mean(results)


def main():

    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default="yes_no", help='Input path of the folder containing the audio files')
    
    # preprocessing parameters (for the MFCC_fast)
    parser.add_argument('-l', '--frame_length', type=int, default=8, help='Frame length')
    parser.add_argument('-s','--frame_step', type=int, default=4, help='Frame step')
    parser.add_argument('-b','--num_mel_bins', type=int, default=16, help='Number of mel bins')
    parser.add_argument('-L','--low_freq', type=int, default=20, help='Lower frequency for the MFCCs')
    parser.add_argument('-U','--up_freq', type=int, default=4000, help='Upper frequency for the MFCCs')
    parser.add_argument('-c','--num_coefficients', default=10, type=int, help='Number of coefficients for the MFCCs')
    parser.add_argument('-r','--rate', default=8000, type=int, help='Resampling rate')

    args = parser.parse_args()

    mfccs_slow = get_mfcc(
        True,
        args.input,
        frame_length=16,
        frame_step=8,
        num_mel_bins=40,
        low_freq=20,
        up_freq=4000,
        num_coefficients=10,
        rate=16000
    )

    mfccs_fast = get_mfcc(
        False,
        args.input, 
        frame_length=args.frame_length, 
        frame_step=args.frame_step, 
        num_mel_bins = args.num_mel_bins, 
        low_freq = args.low_freq, 
        up_freq = args.up_freq, 
        num_coefficients = args.num_coefficients,
        rate = args.rate
    )

    print(f'SNR = {get_snr(mfccs_slow, mfccs_fast):.2f} dB')


if __name__ == '__main__':
    main()


