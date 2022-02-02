import adafruit_dht
import argparse
import numpy as np
import time
import tensorflow as tf
from board import D4

dht_device = adafruit_dht.DHT11(D4)

while True:
    for i in range(7):
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        print(temperature, humidity)

        time.sleep(0.2)
