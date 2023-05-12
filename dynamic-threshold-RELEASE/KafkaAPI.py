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
                self.key = config_dict["key"]
                self.packets_dict = {}
                for field in config_dict["fields"]:
                    self.packets_dict[field] = []
        except FileNotFoundError:
            raise FileNotFoundError
        except KeyError:
            raise KeyError
        else:
            print('配置文件读取成功')

    def packets_append(self, packet: dict):
        if "timestamp" not in packet:
            raise KeyError("Timestamp field must in packet.")
        for key in self.packets_dict:
            if key not in packet:
                self.packets_dict[key].append(np.NaN)
                continue
            self.packets_dict[key].append(packet[key])

    def packets_clear(self):
        for each in self.packets_dict.values():
            each.clear()

    def update(self, mean=True):
        # 需要做的只是把最近的数据拉下来，求平均，打包，而已。
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server], group_id=self.group_id,
                                 auto_offset_reset='earliest')
        messages = consumer.poll(timeout_ms=5000, max_records=100)
        logger = logging.getLogger('dynamic-threshold')
        if messages:
            for records in messages.values():
                for record in records:
                    v = record.value.decode('utf-8')
                    packet = json.loads(v)
                    self.packets_append(packet)
                    logger.debug("接收到新记录: %s" % v)
            logger.info('数据更新成功，最新记录时间为: %s' % (self.packets_dict['timestamp'][-1]))
        else:
            for key in self.packets_dict:
                self.packets_dict[key].append(np.NaN)
            logger.info('无新数据，已填补空值')
        df = pd.DataFrame(self.packets_dict)
        self.packets_clear()
        df.drop("timestamp", axis=1, inplace=True)
        if mean:
            return df.mean(axis=0)
        else:
            return df

    def get_data_after(self, start_time: pd.Timestamp):
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server], group_id=self.group_id,
                                 auto_offset_reset='earliest')
        messages = consumer.poll(timeout_ms=5000, max_records=100)
        logger = logging.getLogger('dynamic-threshold')
        print(messages)
        while len(messages) > 0:
            for records in messages.values():
                for record in records:
                    packet = json.loads(record.value.decode("utf-8"))
                    logger.debug("接收到新记录: %s" % record.value.decode("utf-8"))
                    if pd.Timestamp(packet["timestamp"]) < start_time:
                        continue
                    self.packets_append(packet)
            messages = consumer.poll(timeout_ms=5000, max_records=100)
        df = pd.DataFrame(self.packets_dict)
        if len(self.packets_dict['timestamp']) != 0:
            logger.info('拉取全部数据成功，最新记录时间为: %s' % (self.packets_dict['timestamp'][-1]))
        else:
            logger.warning('自起始时间%s后并无新数据，请检查配置' % start_time)
        self.packets_clear()
        df.index = pd.to_datetime(df["timestamp"])
        df.drop("timestamp", axis=1, inplace=True)
        return df


class KafkaReader1:
    def __init__(self, jconfig: str):
        try:
            with open(jconfig, 'r', encoding="utf-8") as f:
                config_dict = json.load(f)
                self.server = config_dict["server"]
                self.topic = config_dict["topic"]
                self.group_id = config_dict["group-id"]
                self.key = config_dict["key"]
                self.packets_dict = {}
                for field in config_dict["fields"]:
                    self.packets_dict[field] = []
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
                                     auto_offset_reset='earliest',
                                     enable_auto_commit=True)
        except errors.KafkaError as e:
            logger.warning(str(e))
        else:
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
                        row_dict[device] = pd.DataFrame(record, index=[pd.DatetimeIndex(timestamp)]).drop('valid')
                yield row_dict


class PredictDataSaver:
    def __init__(self):
        self.default_path = "config-for-pds.json"
        try:
            with open(self.default_path, 'r', encoding="utf-8") as f:
                self.config = json.load(f)
                self.host = self.config['db-host']
                self.port = self.config['db-port']
                self.user = self.config['db-username']
                self.passwd = self.config['db-password']
                self.db_name = self.config['db-name']
        except FileNotFoundError:
            raise FileNotFoundError("File '%s' is not found\n" % self.default_path)
        except KeyError:
            raise KeyError("Missing important key in config file\n")
        else:
            logger = logging.getLogger('dynamic-threshold')
            log_dir = './log/'
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            file_handler = logging.FileHandler(log_dir + 'pds.log', encoding='utf-8')
            console_handler = logging.StreamHandler()
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel('INFO')

    def save_to_mysql(self, device, dataframe:pd.DataFrame):
        logger = logging.getLogger('kafka_mysql')
        mydb = mysql.connector.connect(
            host=self.host,  # 数据库主机地址
            port=self.port,  # 数据库端口
            user=self.user,  # 数据库用户名
            passwd=self.passwd,  # 数据库密码
            database=self.db_name,  # DB name
            auth_plugin='mysql_native_password'
        )
        my_cursor = mydb.cursor()
        cols = str(tuple(self.config['fields'])).replace("\'", "")
        places = ["%s" for i in range(len(self.config['fields']))]
        places = str(tuple(places)).replace("\'", "")
        sql = "INSERT INTO %s.%s %sVALUES %s" % (self.config['db-name'], self.config['db-table'], cols, places)
        # 做时区转换，数据库使用的时区为UTC
        dataframe['timestamp'] = pd.Timestamp(dataframe['timestamp']) - pd.Timedelta('8h')

        values = tuple([dataframe[key] if key in dataframe else np.NaN for key in self.config['fields']])

        my_cursor.execute(sql, values)
        my_cursor.executemany(sql,)
        mydb.commit()
        mydb.close()
        logger.info("Execute SQL successfully:%s" % (sql % values))






