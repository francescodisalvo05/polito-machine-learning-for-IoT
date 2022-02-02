import json
import sys
import time

from datetime import datetime
from DoSomething import DoSomething

class Subscriber(DoSomething):
    def notify(self, topic, msg):
        
        body = json.loads(msg)
        timestamp = body["bt"]
        events = body['e']
            
        for e in events:
            if e['n'] == 'temperature_exp':
                temperature_exp = e['v']
                temperature_unit = e['u']
            elif e['n'] == 'temperature_pred':
                temperature_pred = e['v']
            elif e['n'] == 'humidity_exp':
                humidity_exp = e['v']
                humidity_unit = e['u']
            elif e['n'] == 'humidity_pred':
                humidity_pred = e['v']

        date = datetime.fromtimestamp(timestamp)
            
        if temperature_pred and temperature_exp:
            
            alert_date = "({:02}/{:02}/{:04} {:02}:{:02}:{:02}) ".format(date.day, date.month, date.year,
                                                                         date.hour, date.minute, date.second)

            alert = "Temperature Alert: Predicted = {:.1f}{} Actual = {:.1f}{}".format(float(temperature_pred), temperature_unit,
                                                                               float(temperature_exp), temperature_unit)
            print(alert_date + alert)

            if humidity_pred and humidity_exp:
                
                alert = "Humidity Alert: Predicted = {:.1f}{} Actual = {:.1f}{}\n".format(float(humidity_pred), humidity_unit,
                                                                                        float(humidity_exp), humidity_unit)
                print(alert_date + alert)



if __name__ == "__main__":
    
    test = Subscriber("Subscriber 2")
    test.run()
    test.myMqttClient.mySubscribe("/282418/alerts")

    while True:
        time.sleep(1)