# kafka-test 2023/5/4 15:28

from kafka import KafkaConsumer

# 以下是Kafka消费者的两种写法。
# 关于Kafka消费者的具体知识，请自行搜索Kafka消费者相关内容，重点学习kafkaConsumer的topic、group_id、auto_offset_reset三个参数的含义
# 第一种写法，读取一次数据，使用consumer.poll()方法，可以设定读取的记录数、以及超时时间
# 自行理解messages的数据格式
try:
    consumer = KafkaConsumer('fish-keeper-raw', bootstrap_servers=['192.168.46.134:9092'], group_id='1-7',
                             auto_offset_reset='earliest')
    messages = consumer.poll(timeout_ms=5000, max_records=100)
    print(messages)
    for records in messages.values():
        for record in records:
            print(record)
            print(record.value.decode("utf-8"))
except Exception as e:
    print("error")

# 第二种写法，循环读取，可以不断获取kafka数据流中的数据

try:
    consumer = KafkaConsumer('fish-keeper-raw', bootstrap_servers=['192.168.46.134:9092'], group_id='1-8',
                             auto_offset_reset='earliest')
    for record in consumer:
        print(record.value.decode("utf-8"))
except Exception as e:
    print("error")

# 特别说明，如果不更换group_id，数据被读取后就无法再次读取
# 为了调试的方便，可以通过更换group_id的方法来从头重新获取新的数据，具体的原理请自行查询
# 建议group_id可以使用当前时间，比如利用pd.Timestamp.now()来获取当前时间戳作为group-id，可以有效避免group-id重复

