# 作者 Ajex
# 创建时间 2023/4/20 20:31
# 文件名 main.py
import pandas as pd
import KafkaAPI
from Processor import Processor
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import env

times = 0
df = None
last_time = None
reader = KafkaAPI.KafkaReader('reader-config.json')
writer = KafkaAPI.KafkaWriter('writer-config.json')


def main_job():
    # 这个函数需要给出预测值
    global df, last_time
    logger = logging.getLogger('dynamic-threshold')
    if times == 0:
        ret = reader.get_data_after(env.valid_time())
        df = ret.resample(env.get_interval()).mean()
        logger.info("数据初始化成功")
        last_time = df.index[-1]
    else:
        new_row = reader.update()
        last_time = last_time + env.get_interval_timedelta()
        new_row.index = [last_time]
        df.append(new_row)
        # 删除过期的数据
        mask = df.index >= env.valid_time()
        df = df.loc[mask]
        logger.info("过期数据已删除，当前有效时间为: %s"%str(env.valid_time()))
    psr = Processor(df)
    steps = env.get_predict_steps()
    predict = psr.predict_with_CEEMDAN(steps)
    predict_json = predict.to_json(date_format='epoch')
    logger.info("已生成预测数据，json格式为: %s"%predict_json)
    writer.write(predict_json)

def main():
    main_job()
    scheduler = BlockingScheduler()
    scheduler.add_job(main_job, 'interval', seconds=env.get_interval_seconds())
    scheduler.start()


if __name__ == '__main__':
    main()
