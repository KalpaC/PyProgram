# 作者 Ajex
# 创建时间 2023/4/20 21:14
# 文件名 env.py
import json
import logging
import pandas as pd
import os

# 读入环境配置文件
try:
    with open("env-config.json", 'r', encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("File 'env-config.json' is not found\n")

# 创建日志
logger = logging.getLogger('dynamic-threshold')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
now = str(pd.Timestamp.now().date())
log_dir = './log'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_file = log_dir + '/' + now + '.log'
f = logging.FileHandler(log_file, encoding='utf-8')
f.setFormatter(formatter)
logger.addHandler(f)
c = logging.StreamHandler()
c.setFormatter(formatter)
logger.addHandler(c)


def get_predict_steps():
    return config['steps']

def get_basic_data_amount():
    return get_predict_steps()*5

def get_buff_size():
    return config['buff-size']

