import cherrypy
import json
import os
import base64
import datetime
import tensorflow as tf
import numpy as np
import time

from cherrypy.process.wspbus import ChannelFailures


class SlowService(object):
    
    exposed = True
    
    def __init__(self):
        # tflite settings
        self.interpreter = tf.lite.Interpreter(model_path='./kws_dscnn_True.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # audio settings
        self.sampling_rate = 16000
        self.frame_length = 640
        self.frame_step = 320
        self.lower_frequency = 20
        self.upper_frequency = 4000
        self.num_mel_bins = 40
        self.num_coefficients = 10
        self.num_spectrogram_bins = (self.frame_length) // 2 + 1
        self.num_frames = (self.sampling_rate - self.frame_length) // self.frame_step + 1
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, 
                                                                                 self.num_spectrogram_bins, 
                                                                                 self.sampling_rate,
                                                                                 self.lower_frequency, 
                                                                                 self.upper_frequency)
    def get_mfcc(self, audio_bytes):
        '''Get the mfcc'''
        
        # decode and normalize
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)

        # padding
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        # STFT
        stft = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        # MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfcc = mfcc[..., :10]

        mfcc = tf.expand_dims(mfcc, -1)  # add channel dimension
        mfcc = tf.expand_dims(mfcc, 0)  # add batch dimension
        
        return mfcc
    
   
    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body['e']

        for event in events:
            if event['n'] == 'audio':
                audio_string = event['vd']

        if audio_string is None:
            raise cherrypy.HTTPError(400, "No audio event")
        
        # decode base64 audio string and load mfcc
        audio_bytes = base64.b64decode(audio_string)
        mfccs = self.get_mfcc(audio_bytes)

        input_tensor = mfccs

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        y_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        y_pred = y_pred.squeeze()
        y_pred = np.argmax(y_pred)

        output_body = json.dumps({'slow_prediction': str(y_pred)})

        return output_body
    
    
    def GET(self, *path, **query):
        pass
    
    
    def POST(self, *path, **query):
        pass
    
    
    def DELETE(self, *path, **query):
        pass

            

if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
        }
    }
    cherrypy.tree.mount(SlowService(), '/predict', conf)

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 1026})
    cherrypy.engine.start()
    cherrypy.engine.block()