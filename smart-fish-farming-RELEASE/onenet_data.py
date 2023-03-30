# onenet_data 2023/1/9 19:14

from sensor_device import *
import cmcc_onenet_api
import device_config
import time
import pandas as pd
import csv

f = open('DeviceData2.csv', 'w+',)
writer = csv.writer(f)
header = ['light', 'temperature', 'humidity']
writer.writerow(header)
while True:
    devices: list[SensorDevice] = device_config.get_devices_from_xml('./device-config.xml')
    d = devices[0]
    cmcc_onenet_api.get_device_payload(d)
    l = [d.light, d.temperature, d.humidity]
    writer.writerow(l)
    print(l)
    f.flush()
    time.sleep(5)
