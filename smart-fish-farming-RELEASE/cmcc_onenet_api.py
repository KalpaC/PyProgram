#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2022/10/11 16:40
# file: cmcc_onenet_api.py.py
# author: Yusheng Wang
# email: yasenwang@bupt.edu.cn
import json
import requests

from sensor_device import SensorDevice


def get_device_data(device_id, api_key):
    url = f'https://api.heclouds.com/devices/{device_id}'
    headers = {
        'api-key': f'{api_key}'
    }
    response = requests.get(url, headers=headers)
    return response.text


def get_device_payload(device: SensorDevice):
    url = f'https://api.heclouds.com/devices/{device.id}/datapoints?datastream_id=Temperature,Humidity,Light&limit=1'
    headers = {
        'api-key': f'{device.key}'
    }
    response = requests.get(url, headers=headers)
    datastreams = json.loads(response.text)['data']['datastreams']  # It is type of list.
    for datastream in datastreams:
        if datastream['id'] == 'Temperature':
            device.temperature = datastream['datapoints'][0]['value']
        if datastream['id'] == 'Light':
            device.light = datastream['datapoints'][0]['value']
        if datastream['id'] == 'Humidity':
            device.humidity = datastream['datapoints'][0]['value']


if __name__ == '__main__':
    pass
