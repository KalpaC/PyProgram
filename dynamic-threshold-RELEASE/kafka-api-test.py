# kafka-api-test 2023/5/9 10:00

import KafkaAPI
import env

reader = KafkaAPI.KafkaReader('reader-config.json')

print(env.valid_time())
df = reader.get_data_after(env.valid_time())

df.to_csv('test.csv')
