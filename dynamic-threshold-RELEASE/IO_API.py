# 作者 Ajex
# 创建时间 2023/4/20 19:01
# 文件名 KafkaReader.py
import json
import logging
import time
import os
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer, errors
import mysql.connector


# 定时拉取数据，而非一有数据就拉取，如果没有拉到数据则插入NaN。
# 每次拉取所有可拉取的数据，然后对其所有数据做平均，从而屏蔽设备、时间间隔的干扰


class KafkaReader:
    def __init__(self, jconfig: str):
        try:
            with open(jconfig, 'r', encoding="utf-8") as f:
                config_dict = json.load(f)
                self.server = config_dict["server"]
                self.topic = config_dict["topic"]
                self.group_id = config_dict["group-id"]
        except FileNotFoundError:
            raise FileNotFoundError
        except KeyError:
            raise KeyError
        else:
            print('配置文件读取成功')

    def timing_records(self):
        logger = logging.getLogger('dynamic-threshold')
        try:
            consumer = KafkaConsumer(self.topic,
                                     bootstrap_servers=[self.server], group_id=self.group_id,
                                     auto_offset_reset='earliest')
        except errors.KafkaError as e:
            logger.warning(str(e))
        else:
            logger.info(
                "Begin to poll records from sever:%s, topic:%s, group-id:%s" % (self.server, self.topic, self.group_id))
            for message in consumer:
                message = message.value.decode('utf-8')
                packet = json.loads(message)
                timestamp = packet['timestamp']
                records = packet['records']
                row_dict = {}
                for device in records:
                    record = records[device]
                    if (not record['valid']) or record['valid'] == 0:
                        row_dict[device] = None
                    else:
                        record.pop('valid')
                        record['timestamp'] = timestamp
                        row_dict[device] = record
                logger.info("Receive data: %s" % str(row_dict))
                yield row_dict
                consumer.commit()


class PredictDataSaver:
    def __init__(self, jconfig: str = None):
        self.default_path = "config-for-pds.json"
        if jconfig:
            self.default_path = jconfig
        try:
            with open(self.default_path, 'r', encoding="utf-8") as f:
                self.config = json.load(f)
                self.host = self.config['db-host']
                self.port = self.config['db-port']
                self.user = self.config['db-username']
                self.passwd = self.config['db-password']
                self.db_name = self.config['db-name']
                self.db_table = self.config['db-table']
        except FileNotFoundError:
            raise FileNotFoundError("File '%s' is not found\n" % self.default_path)
        except KeyError:
            raise KeyError("Missing important key in config file\n")

    def save_to_mysql(self, device, dataframe: pd.DataFrame):
        mydb = mysql.connector.connect(
            host=self.host,  # 数据库主机地址
            port=self.port,  # 数据库端口
            user=self.user,  # 数据库用户名
            passwd=self.passwd,  # 数据库密码
            database=self.db_name,  # DB name
            auth_plugin='mysql_native_password'
        )
        my_cursor = mydb.cursor()
        keys = tuple(dataframe.keys().tolist() + ['name'])
        cols = str(keys).replace('\'', '')
        places = ['%s' for i in range(len(keys))]
        places = str(tuple(places)).replace('\'', '')
        sql = 'INSERT %s.%s %s VALUES %s' % (self.db_name, self.db_table, cols, places)
        print(sql)
        rows = []
        for i in range(len(dataframe)):
            row = dataframe.iloc[i].tolist()
            row.append(device)
            rows.append(row)
        print(rows)
        my_cursor.executemany(sql, rows)
        mydb.commit()
        mydb.close()

