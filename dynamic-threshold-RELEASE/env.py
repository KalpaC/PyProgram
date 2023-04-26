# 作者 Ajex
# 创建时间 2023/4/20 21:14
# 文件名 env.py
import json

import pandas as pd

try:
    with open("env-config.json",'r',encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("File 'env-config.json' is not found\n")

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


