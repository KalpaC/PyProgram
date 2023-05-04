# 作者 Ajex
# 创建时间 2023/4/20 19:01
# 文件名 KafkaReader.py
import json
import logging

import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer,errors


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

    def packets_append(self, packet: dict):
        if "Timestamp" not in packet:
            raise KeyError("Timestamp field must in packet.")
        for key in self.packets_dict:
            if key not in packet:
                self.packets_dict[key].append(np.NaN)
                continue
            self.packets_dict[key].append(packet[key])

    def packets_clear(self):
        for each in self.packets_dict:
            each.clear()

    def update(self,mean=True):
        # 需要做的只是把最近的数据拉下来，求平均，打包，而已。
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server], group_id=self.group_id)
        messages = consumer.poll(timeout_ms=5000, max_records=100)
        logger = logging.getLogger('dynamic-threshold')
        if messages:
            for records in messages.values():
                for record in records:
                    v = record.value.decode('utf-8')
                    packet = json.loads(v)
                    self.packets_append(packet)
                    logger.debug("接收到新记录: %s"%v)
            logger.info('数据更新成功，最新记录时间为: %s'%(self.packets_dict['Timestamp'][-1]))
        else:
            for key in self.packets_dict:
                self.packets_dict[key].append(np.NaN)
            logger.info('无新数据，已填补空值')
        df = pd.DataFrame(self.packets_dict)
        self.packets_clear()
        df.drop("Timestamp", axis=1, inplace=True)
        if mean:
            return df.mean(axis=0)
        else:
            return df

    def get_data_after(self, start_time: pd.Timestamp):
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server], group_id=self.group_id)
        messages = consumer.poll(timeout_ms=500, max_records=1000)
        logger = logging.getLogger('dynamic-threshold')
        while len(messages) > 0:
            for records in messages.values():
                for record in records:
                    packet = json.loads(record.value.decode("utf-8"))
                    logger.debug("接收到新记录: %s"%record.value.decode("utf-8"))
                    if pd.Timestamp(packet["Timestamp"]) < start_time:
                        continue
                    self.packets_append(packet)
            messages = consumer.poll(timeout_ms=500, max_records=1000)
        df = pd.DataFrame(self.packets_dict)
        logger.info('拉去全部数据成功，最新记录时间为: %s' % (self.packets_dict['Timestamp'][-1]))
        self.packets_clear()
        df.index = pd.to_datetime(df["Timestamp"])
        df.drop("Timestamp", axis=1, inplace=True)
        return df

class KafkaWriter:
    def __init__(self,jconfig:str):
        try:
            with open(jconfig, 'r', encoding="utf-8") as f:
                config_dict = json.load(f)
                self.server = config_dict["server"]
                self.topic = config_dict["topic"]
                self.group_id = config_dict["group-id"]
                self.key = config_dict["key"]
        except FileNotFoundError:
            raise FileNotFoundError
        except KeyError:
            raise KeyError

    def write(self, contend:str):
        producer = KafkaProducer(bootstrap_servers=[self.server])
        logger = logging.getLogger('dynamic-threshold')

        try:
            future = producer.send(self.topic, contend.encode('utf-8'), key=self.key.encode('utf-8'))
            future.get(timeout=3)
            logger.info("Kafka回写成功")
        except errors.KafkaError as e:
            logger.warning("Kafka写入失败")
        producer.flush()
        producer.close()




