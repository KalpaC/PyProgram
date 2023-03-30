# 不用!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2022/10/11 16:52
# file: sensor_device.py
# author: Yusheng Wang
# email: yasenwang@bupt.edu.cn
import json


class SensorDevice:
    def __init__(self):
        self.id = None
        self.key = None
        self.name = 'Default'
        self.humidity = 0.0
        self.temperature = 0.0
        self.light = 0.0

    def set_light(self, light):
        self.light = light
        return self

    def set_temperature(self, temperature):
        self.temperature = temperature
        return self

    def set_humidity(self, humidity):
        self.humidity = humidity
        return self

    def __repr__(self):
        return f'SensorDevice: Name: {self.name} ID: {self.id}\n' \
               f'light: {self.light}, temperature: {self.temperature}, humidity: {self.humidity}'

    def __str__(self):
        return json.dumps({"name": self.name, "id": self.id, "light": self.light,
                           "temperature": self.temperature, "humidity": self.humidity, }, ensure_ascii=False)


print(SensorDevice())
