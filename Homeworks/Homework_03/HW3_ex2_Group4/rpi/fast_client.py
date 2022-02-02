import json
import os
import base64
import requests
import sys

from scipy import signal
import tensorflow as tf
import numpy as np

from signal_generator import SignalGenerator


tf.random.set_seed(42)
np.random.seed(42)


def main():
    tflite_model = os.path.join(".", "kws_dscnn_True.tflite")

    if os.path.exists(tflite_model) is False:
        print(f"Error: tflite not found!")
        sys.exit(1)    
        
    test_files, labels, MFCC_OPTIONS = get_data()    
        
    accuracy, communication_cost = make_inference(test_files, labels, MFCC_OPTIONS, tflite_model)
    
    print('Accuracy = {:.2f} %'.format(accuracy * 100))
    print('Communication cost = {:.2f} MB'.format(communication_cost / 1048576))
    
             
def get_data():
    '''Get the audio files and their labels, together with the 
    used MFCC_OPTIONS.
    
    Returns:
        - test_files (list) : list containing the audio path
        - labels (list) : list of labels
        - MFCC_OPTIONS (dic) : MFCC options used for the audio preprocessing
    '''
    
    dataset_path = os.path.join("data")
    
    
    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                fname='mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir=dataset_path)
    
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800

    labels = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
    
    MFCC_OPTIONS = {'frame_length': 320,
                    'frame_step': 160,
                    'lower_frequency': 20,
                    'upper_frequency': 4000,
                    'num_mel_bins': 16,
                    'num_coefficients': 10,
                    'resampling_rate' : 8000}
    
    return test_files, labels, MFCC_OPTIONS
    
    
def make_inference(test_files, labels, MFCC_OPTIONS, tflite_model):
    '''Make inference on all the audio files, using the success_checker
    if the difference between the probabilities of the two most likely 
    classes is less than 0.2.
    
    Args:
        - test_files (list) : list containing the audio path
        - labels (list) : list of labels
        - MFCC_OPTIONS (dic) : MFCC options used for the audio preprocessing
        - tflite_model (str) : path of the tflite model
    Returns:
        - accuracy (float) : accuracy over all the predictions
        - cum_communication_bytes (float) : total communication cost with 
                                            slow_service.py
    '''
    
    signal_generator = SignalGenerator(labels, 16000, mfcc=True, **MFCC_OPTIONS)    
    
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy, count, cum_communication_bytes = 0, 0, 0.0
    
    threshold_slow_call = 0.2
    
    for i, audio_path in enumerate(test_files):
        
        features, labels = signal_generator.preprocess(audio_path)
        features = tf.expand_dims(features, axis=0)
        labels = labels.numpy().squeeze()
        
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index']).squeeze()
        predictions = tf.nn.softmax(tf.convert_to_tensor(predictions)).numpy() # hint from the Professor
            
        second_pred, first_pred = np.argsort(predictions)[-2:]  # get index 
        second_prob, first_prob = predictions[[second_pred, first_pred]] # get probability
        
        if first_prob - second_prob < threshold_slow_call:
            audio_binary = tf.io.read_file(audio_path)
            predicted_label, curr_communication_cost = success_checker(first_prob, second_prob, first_pred, audio_binary)
            cum_communication_bytes += curr_communication_cost
        else:
            predicted_label = first_pred
        
        if predicted_label == labels:
            accuracy += 1

        count += 1
            
            
    return accuracy / (count * 1.0), cum_communication_bytes


def success_checker(first_prob, second_prob, first_prediction, audio_binary):
    '''It will ask for the prediction of slow_service.py
    
    Args:
        - first_prob (float) : probability of the most likely predicted class
        - second_prob (float) : probability of the second most likely predicted class
        - first_prediction (str) : predicted class
        - audio_binary (str) : base64 string of the audio file
        
    Returns:
        - prediction (str) : prediction of the slow service
        - len_senml (float) : length of the senml message (i.e. communication cost)
    
    '''
    
    audio_b64bytes = base64.b64encode(audio_binary.numpy())
    audio_string = audio_b64bytes.decode()

    body_request = {"bn": "http://0.0.0.0:1026/",
                    "e": [{"n": "audio", "u": "/", "t": "0", "vd": audio_string}]}

    url = "http://0.0.0.0:1026/predict"
    response = requests.put(url, json=body_request)       

    if response.status_code == 200:
        
        body_response = response.json()
        prediction = int(body_response['slow_prediction'])
        
        senml = json.dumps(body_request)
        len_senml = len(senml)
        
        return prediction, len_senml
    
    else:
        
        print(response.status_code)
        print("")
        print(response.text)
        sys.exit(1)
    

    

if __name__ == '__main__':
    main()