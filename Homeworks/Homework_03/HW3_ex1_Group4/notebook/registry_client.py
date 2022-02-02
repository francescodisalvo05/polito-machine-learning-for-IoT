import base64
import requests
import json
import datetime
import sys 

from monitoring_client import Subscriber

    
######################
## /add model
######################
        
models = ['cnn.tflite','mlp.tflite']

# upload both models
for model in models:

    with open(model, 'rb') as tflite_model:
        binary_tflite = tflite_model.read()
        base64_tflite = base64.b64encode(binary_tflite)
        base64_message = base64_tflite.decode('utf-8')

    url_add = "http://0.0.0.0:8080/add"

    json_add = {'name':model, 'model': base64_message}
    request_add = requests.put(url_add, json=json_add)

    if request_add.status_code != 200:
        print(request_add.status_code)
        if request_add.status_code == 404:
            print('Wrong path!')
        else:
            print(request_add.text)
        
        
        sys.exit(1)



######################
## /list models
######################

url_list = "http://0.0.0.0:8080/list"
request_list = requests.get(url_list)

if request_list.status_code == 200:
    if len(request_list.json()['models']) != 2:
        print('There are not two models!')
        sys.exit(1)
else:
    print(request_list.status_code)
    if request_list.status_code == 404:
            print('Wrong path!')
    else:
        print(request_list.text)
    sys.exit(1)



######################
## /predict
######################
selected_model = 'mlp.tflite' 

tthres, hthres = 0.1, 0.2

url_predict = "http://0.0.0.0:8080/predict?model={}&tthres={}&hthres={}".format(selected_model, tthres, hthres)
request_predict = requests.post(url_predict)

if request_predict.status_code != 200:
    print(request_predict.status_code)
    if request_predict.status_code == 404:
            print('Wrong path!')
    else:
        print(request_predict.text)
        
