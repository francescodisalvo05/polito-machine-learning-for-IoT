import numpy as np

import time
from datetime import datetime
import os

from argparse import ArgumentParser

import tensorflow as tf


def normalize(col_temp, col_hum):

    # from the datasheet

    min_temp, max_temp = 0.0, 50.0
    min_hum, max_hum = 20.0, 90.0

    col_temp_norm = (col_temp.astype('float') ) / (max_temp - min_temp)
    col_hum_norm = (- min_hum + col_hum.astype('float') ) / (max_hum - min_hum)

    return col_temp_norm, col_hum_norm


def print_statistics(output_path, start, end):

    # time = (end-start) * 1000.0
    # print(f'Total time = {time:.3f} ms')

    output_size = os.path.getsize(output_path)
    print(f'Final size = {output_size} B')


def make_dataset(args):
    
    dates = np.array([])
    hours = np.array([])
    temps = np.array([])
    hums = np.array([])
    
    # open the csv file without using pandas 
    # in order to speedup the process
    with open(args.input,"r") as file:
        csv_reader = file.readlines()
        
        for row in csv_reader:
            date, hour, temp, hum = row.strip().split(",")
            dates = np.append(dates,date)
            hours = np.append(hours,hour)
            temps = np.append(temps,temp)
            hums = np.append(hums,hum)

    # normalize, if requested 
    if args.normalize:
        temps, hums = normalize(temps, hums)


    with tf.io.TFRecordWriter(args.output) as writer:
        for date, hour, temp, hum in zip(dates, hours, temps, hums):
            # join both date and hour
            raw_date = ",".join([date,hour])
            date = datetime.strptime(raw_date, '%d/%m/%Y,%H:%M:%S')
            # convert it into a posix format
            posix_date = time.mktime(date.timetuple())

            # create datetime feature
            datetime_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_date)]))
	   
            # if we have to normalize, use Float representation, otherwise Integer
            if not args.normalize:
                temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temp.astype(np.int64)]))
                humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[hum.astype(np.int64)]))
            else:
                temperature = tf.train.Feature(float_list=tf.train.FloatList(value=[temp]))
                humidity = tf.train.Feature(float_list=tf.train.FloatList(value=[hum]))

            mapping = {'datetime': datetime_feature,
                       'temperature': temperature,
                       'humidity': humidity,
                       }
            # append the new element in the dataset
            final = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(final.SerializeToString())


def main():

    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default="dataset_ex1.csv", help='Set the input path')
    parser.add_argument('--output',type=str, default="dataset_output.tfrecord", help='Set the output path')
    parser.add_argument('--normalize', action='store_true', help='Normalize the data')

    args = parser.parse_args()

    start = time.time()

    make_dataset(args)

    end = time.time()

    print_statistics(args.output, start, end)


if __name__ == '__main__':

    main()
