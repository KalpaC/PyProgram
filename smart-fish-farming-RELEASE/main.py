# 不用这行!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2022/10/11 16:39
# file: main.py.py
# author: Yusheng Wang
# email: yasenwang@bupt.edu.cn
from kafka import KafkaProducer
from apscheduler.schedulers.blocking import BlockingScheduler
from sensor_device import *

import cmcc_onenet_api
import device_config


KAFKA_TOPIC = 'smart-fish-farming'
TOPIC_KEY = b'previous_data'


def main_job():
    # Initial Kafka producer.
    producer = KafkaProducer(bootstrap_servers=['192.168.36.139:9092'])
    # Load one-net devices.
    devices:list[SensorDevice] = device_config.get_devices_from_xml('./device-config.xml')

    for device in devices:
        cmcc_onenet_api.get_device_payload(device)
        future = producer.send(KAFKA_TOPIC, str(device).encode('utf-8'), key=TOPIC_KEY)
        future.get(timeout=10)

    producer.flush()
    producer.close()


def main():
    scheduler = BlockingScheduler()
    scheduler.add_job(main_job, 'interval', seconds=10)
    scheduler.start()


if __name__ == '__main__':
    main()
