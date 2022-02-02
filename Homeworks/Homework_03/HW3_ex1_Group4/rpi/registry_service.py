import cherrypy
import json
import os
import base64
import time
import adafruit_dht

from DoSomething import DoSomething
from board import D4
import tensorflow as tf
import numpy as np
from datetime import datetime


class ModelRegistryAdd:
    '''Store a tflite model in a local subfolder called models 
        located at the same path of the service script.
        
        Request: 
            - Path : /add
            - Parameters : any
            - Body :
                model: (base64 str) the tflite model
                name: (str) the model name -- I assume with .tflite (ask!)
        Response:
            - Code : 200 (if successful) or 400 (otherwise)
            - Body : None
    '''
    exposed = True
    
    def __init__(self):
        self.model_path = './models/'
    
    
    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        
        body_keys = list(input_body.keys())
        
        base64_tflite = input_body['model'].encode('utf-8')
        model_name = input_body['name']

        # check the body
        if not model_name:
            raise cherrypy.HTTPError(400, "Missing model name!")
        if not base64_tflite:
            raise cherrypy.HTTPError(400, "Missing tflite model!")
                
        # create the folder if it doesn't exists
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        
        # write the tflite model on disk
        # overwrite if the file exists
        with open(self.model_path + model_name, 'wb') as tflite_model:
            decoded_tflite = base64.decodebytes(base64_tflite)
            tflite_model.write(decoded_tflite)
          
        return
    
    
    def GET(self, *path, **query):
        pass
    

    def POST(self, *path, **query):
        pass
    
    
    def DELETE(self, *path, **query):
        pass


class ModelRegistryList:
    '''List the names all the models stored in the models folder.
        
        Request: 
            - Path : /list
            - Parameters : any
            - Body : any
        Response:
            - Code : 200 (if successful) or 400 (otherwise)
            - Body : {models : list of strings}
        '''
    
    exposed = True
    
    def __init__(self):
        self.model_path = './models/'
        
    
    def GET(self, *path, **query):
        # get model names from the directory
        model_names = os.listdir(self.model_path)
        
        if len(model_names) == 0:
            raise cherrypy.HTTPError(400,"Empty folder!")
        
        # remove the extentions
        model_names = [m for m in model_names if m.split(".")[1] == 'tflite']
    
        output = {'models' : model_names}
        output_json = json.dumps(output)

        return output_json

    
    def POST(self, *path, **query):
        pass  
            
    
    def PUT(self, *path, **query):
        pass
    
    
    def DELETE(self, *path, **query):
        pass


class ModelRegistryPredict:
    '''Measure temperature and humidity values with the DHT-11 sensor every 1s. 
    As soon as 6 samples are measured, make a prediction with the model 
    indicated in the request every 1s. Compare the measured vs. the predicted 
    value. If the prediction error (absolute error) of temperature (humidity) 
    is greater than a temperature (humidity) threshold, send an alert to a single 
    or multiple remote clients using the SenML+JSON format.
    
    Request: 
        - Path : /predict
        - Parameters : 
            model: (str) the name of the model
            tthres: (float) the temperature threshold in °C 
            hthres: (float) the humidity threshold in %
        - Body : any
    Response:
        - Code : 200 (if successful) or 400 (otherwise)
        - Body : any
    '''
    exposed = True
    
    def __init__(self):
        self.model_path = './models/'
        
        
    
    def POST(self, *path, **query):
        # check query
        if len(query) != 3:
            raise cherrypy.HTTPError(400, "Wrong query!")
        
        model, tthres, hthres = query.get('model'), query.get('tthres'), query.get('hthres')
        
        # notify for each missing value
        if not model:
            raise cherrypy.HTTPError(400, "Missing model!")
            
        if not tthres:
            raise cherrypy.HTTPError(400, "Missing ttresh!")       
        
        if not hthres:
            raise cherrypy.HTTPError(400, "Missing htresh!")
            
        publisher = DoSomething("Publisher")
        publisher.run()

        counter = 0
        
        dht_device = adafruit_dht.DHT11(D4)
        

        while True:    

            now = datetime.now()
            timestamp = int(now.timestamp())

            interpreter = tf.lite.Interpreter(model_path=self.model_path + model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            window = np.zeros([1, 6, 2], dtype=np.float32)
            expected = np.zeros(2, dtype=np.float32)

            MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
            STD = np.array([8.654227, 16.557089], dtype=np.float32)
            
            
            # get new recordings
            try:
                temperature = dht_device.temperature
                humidity = dht_device.humidity
            except RuntimeError as error:
                print(error.args[0])
                
    
            # fill the first window
            if counter < 6:
                window[0, counter, 0] = np.float32(temperature)
                window[0, counter, 1] = np.float32(humidity)

            else:

                expected[0] = np.float32(temperature)
                expected[1] = np.float32(humidity)

                window = (window - MEAN) / STD
                interpreter.set_tensor(input_details[0]['index'], window)
                interpreter.invoke()
                predicted = interpreter.get_tensor(output_details[0]['index'])

                exp_temp, exp_hum = expected[0], expected[1]
                pred_temp, pred_hum = predicted[0, 0], predicted[0, 1]

                # shift the window
                window[:, 0:5, :] = window[:, 1:6, :]
                window[:, -1, 0] = exp_temp
                window[:, -1, 1] = exp_hum

                events = []

                tthres, hthres = float(query.get('tthres')), float(query.get('hthres'))

                if abs(exp_temp - pred_temp) > tthres:
                    events.append({"n": "temperature_exp", "u": "°Cel", "t": 0, "v": str(exp_temp)})
                    events.append({"n": "temperature_pred", "u": "°Cel", "t": 0, "v": str(pred_temp)})
                
                if abs(exp_hum - pred_hum) > hthres:
                    events.append({"n": "humidity_exp", "u": "%", "t": 0, "v": str(exp_hum)})
                    events.append({"n": "humidity_pred", "u": "%", "t": 0, "v": str(pred_hum)})

                body = {
                    "bn": "http://0.0.0.0:8080", 
                    "bt": timestamp,
                    "e": events
                }

                publisher.myMqttClient.myPublish("/282418/alerts", json.dumps(body))
                time.sleep(1) 

            counter += 1

    
    def GET(self, *path, **query):
        pass

    
    def PUT(self, *path, **query):
        pass
    
    
    def DELETE(self, *path, **query):
        pass


def main():
    conf = {"/":{"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}

    cherrypy.tree.mount(ModelRegistryList(), "/list", conf)
    cherrypy.tree.mount(ModelRegistryAdd(), "/add", conf)
    cherrypy.tree.mount(ModelRegistryPredict(), "/predict", conf)

    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.config.update({"server.socket_port": 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()    


if __name__ == "__main__":
    main()