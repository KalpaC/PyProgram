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


def valid_timedelta():
    et = pd.Timedelta(config['expiration-time'])
    return et


def valid_time():
    # 用现在的时间和有效期就能算出了
    now = pd.Timestamp.now()
    et = valid_timedelta()
    return now - et


def get_interval():
    return config['interval']


def get_interval_timedelta():
    return pd.Timedelta(config['interval'])


def get_predict_steps():
    return config['steps']


def get_interval_seconds():
    d = get_interval_timedelta().seconds
    return d
