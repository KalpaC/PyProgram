# 作者 Ajex
# 创建时间 2023/4/7 16:24
# 文件名 KafkaAPI.py
from kafka import *
import pandas as pd
import json

from kafka.consumer.fetcher import ConsumerRecord
SUCCEED = 0
FILE_NOT_FOUND = -1
CONFIG_ERROR = -2
OTHER_ERROR = -3

def LoadKafkaReader():
    try:
        JsonConfig = "Kafka-API-Config.json"
        with open(JsonConfig,'r',encoding="utf-8") as f:
            config = json.load(f)
            topic = config["basic"]['topic']
            server = config["basic"]['server']
            first_size = config["basic"]['first_size']
            update_size = config["basic"]['update_size']
            delay = config["basic"]['delay']
            empty_dict = config["KafkaRecordDict"]
            return KafkaReader(topic,server,first_size,update_size,delay,empty_dict),SUCCEED
    except FileNotFoundError:
        return None, FILE_NOT_FOUND
    except KeyError:
        return None,CONFIG_ERROR
    except Exception:
        return None,OTHER_ERROR

def get_data(messages,packet_dict):
    """
    直接处理poll方法返回的字典
    :param messages: 字典
    :param packet_dict: 数据写入的缓冲区
    :return: NULL
    """
    # kafka poll 结果返回的字典格式为{Partition():[ConsumerRecord]}
    for records in messages.values():
        # 对每个分区的字典下的Record列表
        for record in records:
            # 对每个列表中的每个Record
            data_dict = json.loads(record.value)
            for key in packet_dict:
                if key not in data_dict:
                    # 缺失项，应填补
                    packet_dict[key].append(None)
                    continue
                packet_dict[key].append(data_dict[key])
    pass

class KafkaReader:
    # 获取consumer，定时从consumer中获取数据，当数据足够时，打包成df，发送给DataLayer
    # 初步思路是kafka一个死循环拉数据，当数据足够时，打包成一个df，启动一个processor线程，处理这个df，并给出可视化。
    def __init__(self,topic,server,first_size,update_size,delay,empty_record_dict:dict):
        """
        提供了从指定kafka上读取数据的规范方法（kafka_timed_pull）
        :param topic: kafka-topic
        :param server: ip-address:port
        :param delay: wait how many microsecond
        :param first_size: the first returned dataframe's size
        :param update_size: follow-up returned dataframe's size
        """
        self.topic = topic
        self.server = server
        self.delay = delay
        self.first_size = first_size
        self.update_size = update_size
        self.packet_dict = empty_record_dict

    def kafka_timed_pull(self):
        """
        可以直接使用for语句迭代来获取数据
        :return: a generator of dataframe which contain update data
        """
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server],group_id='test')
        counter = 0
        first = True
        other = False
        while True:
            messages = consumer.poll(timeout_ms=self.delay,max_records=1)
            if messages:
                get_data(messages, self.packet_dict)
                counter += 1
                if first and counter == self.first_size:
                    df = pd.DataFrame(self.packet_dict)
                    yield df
                    first = False
                    other = True
                    counter = 0
                    clear_dict_list(self.packet_dict)
                elif other and counter == self.update_size:
                    df = pd.DataFrame(self.packet_dict)
                    yield df
                    counter = 0
                    clear_dict_list(self.packet_dict)

def clear_dict_list(dict_list):
    for each_list in dict_list:
        each_list.clear()
