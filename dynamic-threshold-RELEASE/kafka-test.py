# kafka-test 2023/5/4 15:28

from kafka import KafkaConsumer
try:
    consumer = KafkaConsumer('fish-keeper-raw', bootstrap_servers=['192.168.46.134:9092'], group_id='1')
    for message in consumer:
        print(message.value.decode('utf-8'))
except Exception as e:
    print("error")


