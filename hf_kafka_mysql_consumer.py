#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2022/9/12 10:22
# file: hf_kafka_mysql_consumer.py
# author: Yusheng Wang
# email: yasenwang@bupt.edu.cn

"""
This module is used to keep watch on kafka data streams, and download the data to MySQL.

Detail.

Typical usage example:

"""

import json
import logging

from kafka.consumer import KafkaConsumer
import mysql.connector


def save_to_mysql(stock_code, stock_data):
    logger = logging.getLogger('kafka_mysql')
    mydb = mysql.connector.connect(
        host="192.168.36.26",  # 数据库主机地址
        user="ar_test",  # 数据库用户名
        passwd="Ar_test123",  # 数据库密码
        database="ar",  # DB name
        auth_plugin='mysql_native_password'
    )
    my_cursor = mydb.cursor()

    sql = "INSERT INTO ar.t_stock_realtime (stock_code, stock_data) VALUES (%s, %s)"
    val = (stock_code, stock_data)
    my_cursor.execute(sql, val)
    mydb.commit()
    mydb.close()


def main():
    # Config logging.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('kafka_mysql')
    handler = logging.FileHandler('./kafka_mysql.log')
    logger.addHandler(handler)
    logger.info('====Kafka MySQL consumer START====')

    consumer = KafkaConsumer('ar_previous_data_test',
                             bootstrap_servers=['192.168.36.139:9092'])

    for message in consumer:
        message = message.value
        message = message.decode('utf-8')
        msg = json.loads(message)
        save_to_mysql(msg['stock_code'], message)
        logger.info(msg)

        # consumer.commit()


if __name__ == '__main__':
    main()
