# 作者 Ajex
# 创建时间 2023/4/20 20:31
# 文件名 main.py
import pandas as pd

import KafkaReader as kr
from Processor import Processor
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
import env

times = 0
df = None
last_time = None


def main_job():
    # 这个函数需要给出预测值
    global df, last_time
    reader = kr.KafkaReader('reader-config.json')
    if times == 0:
        ret = reader.get_data_after(env.valid_time())
        df: pd.DataFrame = ret.resample(env.get_interval()).mean()
        last_time = df.index[-1]
    else:
        new_row = reader.update()
        new_row.index = [last_time + env.get_interval_timedelta()]
        df.append(new_row)
        mask = df.index >= env.valid_time()
        df = df.loc[mask]
        psr = Processor(df)
        steps = env.get_predict_steps()
        predict = psr.predict_with_CEEMDAN(steps)
        predict_json = predict.to_json(date_format='epoch')

        # 就差预测了
        # 至少应该给出未来几次的预测值
        # 写入当前时间、未来某几个时间点的时间戳以及预测值到kafka数据流
        # 由kafka-mysql组件统一存入数据库
        # 格式可以是 当前时间戳、未来时间间隔steps个数据的预测值


pass


def main():
    scheduler = BlockingScheduler()
    scheduler.add_job(main_job, 'interval', seconds=5)
    scheduler.start()


if __name__ == '__main__':
    main()
