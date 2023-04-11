# 作者 Ajex
# 创建时间 2023/4/7 16:24
# 文件名 KafkaAPI.py
from kafka import KafkaConsumer
import pandas as pd

def get_data(messages,packet_dict):
    """
    直接处理poll方法返回的字典
    :param messages: 字典
    :param packet_dict: 数据写入的缓冲区
    :return: NULL
    """
    pass

class KafkaReader:
    # 获取consumer，定时从consumer中获取数据，当数据足够时，打包成df，发送给DataLayer
    # 初步思路是kafka一个死循环拉数据，当数据足够时，打包成一个df，启动一个processor线程，处理这个df，并给出可视化。
    def __init__(self,topic,server,first_size,update_size,delay):
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

    def kafka_timed_pull(self):
        """
        可以直接使用for语句迭代来获取数据
        :return: a generator of dataframe which contain update data
        """
        consumer = KafkaConsumer(self.topic, bootstrap_servers=[self.server])
        packet_dict = {}
        counter = 0
        first = True
        other = False
        while True:
            messages = consumer.poll(timeout_ms=self.delay)
            if messages:
                get_data(messages, packet_dict)
                counter += 1
                if first and counter == self.first_size:
                    df = pd.DataFrame(packet_dict)
                    yield df
                    first = False
                    other = True
                    counter = 0
                    packet_dict.clear()
                elif other and counter == self.update_size:
                    df = pd.DataFrame(packet_dict)
                    yield df
                    counter = 0
                    packet_dict.clear()