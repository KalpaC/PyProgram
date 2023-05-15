# final 2023/5/12 17:10
import logging
import time

import pandas as pd

import IO_API
import Processor
import env


def main():
    data = {}
    steps = env.get_predict_steps()
    basic_data_amount = env.get_basic_data_amount()
    kr = IO_API.KafkaReader('reader-config.json')
    pds = IO_API.PredictDataSaver("config-for-pds.json")
    buff_size = env.get_buff_size()
    logger = logging.getLogger('dynamic-threshold')
    for new_data_for_devices in kr.timing_records():
        for device in new_data_for_devices:
            if new_data_for_devices[device] is None:
                if device in data:
                    # 如果是无效数据，说明已经很久没有有效数据了，之前的数据作废
                    data.pop(device)
                continue
            if device not in data:
                data[device] = [new_data_for_devices[device]]
            else:
                if new_data_for_devices[device]['timestamp'] == data[device][-1]['timestamp']:
                    continue
                data[device].append(new_data_for_devices[device])
        for device in data:
            l = len(data[device])
            if l >= basic_data_amount and l % steps == 0:
                df = pd.DataFrame(data[device])
                df.index = pd.DatetimeIndex(df['timestamp'])
                df.drop('timestamp', axis=1, inplace=True)
                print(device, "源数据：\n", df.tail(10))
                psr = Processor.Processor(df)
                predict = psr.predict_with_CEEMDAN(steps + 1)
                print(predict)
                pds.save_to_mysql(device, predict)
                # 生成预测数据之后要将数据发送至sql上
                # 上传的数据应该是无间断的预测数据
                # pds.save_to_mysql(device, predict)
                # try:
                #     psr = Processor.Processor(df)
                #     predict = psr.predict_with_CEEMDAN(steps + 1)
                #     print(predict)
                #     # 生成预测数据之后要将数据发送至sql上
                #     # 上传的数据应该是无间断的预测数据
                #     pds.save_to_mysql(device, predict)
                # except:
                #     print("fail to predict")
            if l > buff_size:
                data[device] = data[device][-basic_data_amount:]
                logger.info("Clean buffer")


if __name__ == '__main__':
    main()
