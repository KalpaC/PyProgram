# WashedDataSaver 2023/5/3 17:13
import json
from kafka import KafkaConsumer
import logging
import mysql.connector
import numpy as np

class WashedDataSaver:
    def __init__(self):
        try:
            with open("config-for-wds.json", 'r', encoding="utf-8") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("File 'env-config.json' is not found\n")

    def main(self):
        logger = logging.getLogger('kafka_mysql')
        handler = logging.FileHandler('wds.log')
        logger.addHandler(handler)
        logger.info('====Kafka MySQL WDS consumer START====')
        consumer = KafkaConsumer(self.config['topic'],
                                 bootstrap_servers=[self.config['kafka-server']], group_id=self.config['group-id'])
        for message in consumer:
            message = message.value
            message = message.decode('utf-8')
            packet = json.loads(message)
            self.save_to_mysql(packet)

    def save_to_mysql(self, packet):
        mydb = mysql.connector.connect(
            host=self.config['db-server'],  # 数据库主机地址
            user=self.config['db-username'],  # 数据库用户名
            passwd=self.config['db-password'],  # 数据库密码
            database=self.config['db-name'],  # DB name
            auth_plugin='mysql_native_password'
        )
        my_cursor = mydb.cursor()
        # 数据库名、数据库表名
        cols = str(tuple(self.config['fields'])).replace("\'","")
        places = ["%d" for i in range(len(self.config['fields']))]
        places[0]="%s"  # 此处默认第一项就是Timestamp
        places = str(tuple(places)).replace("\'","")
        sql = "INSERT INTO %s.%s %s VALUES %s"%(self.config['db-name'],self.config['db-table'],cols,places)
        values = tuple([packet[key] if key in packet else np.NaN for key in self.config['fields']])
        my_cursor.execute(sql,values)
        mydb.commit()
        mydb.close()

if __name__ == '__main__':
    wds = WashedDataSaver()
    wds.main()
