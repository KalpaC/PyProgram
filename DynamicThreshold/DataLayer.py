import numpy as np
import pandas as pd
import json
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt

config = "config.json"


# 获得数据

#   从数据源获取数据
#   数据处理
# 数据应该包括，时间戳，对应的水温、湿度、光照等数据。
# 数据本身从Kafka上读取得到，但由于目前Kafka已停机，暂时只可规定数据应该表示为DataFrame。
# 实际上Kafka上的数据是流式的，还需将数据汇集并处理成df类型


# 规定外部接口
#   功能1，面向外部数据，做数据格式的转换和处理

class DataLayer:
    # 列名规定见ColName.json
    def __init__(self, df: pd.DataFrame, interval: str = None):
        """
        :param df: 至少含有时间戳与一列数据的dataframe
        :param interval: 设置的重采样间隔（周期）
        """
        # 载入列名配置文件
        self.config = None
        self.loadConfig()
        self.colName = self.getColName()
        self.df = df
        self.resampleDF = None

        # 将TIME一列改为索引，按照时间戳排序
        TIME = self.colName['TIME']
        self.df.index = pd.to_datetime(df[TIME])
        self.df.drop(TIME, axis=1, inplace=True)
        self.df.sort_index(inplace=True)
        # 重采样
        if interval:
            self.interval = interval
            self.resampleDF = self.df.resample(interval).mean()
            NaCnt = self.resampleDF[self.resampleDF.keys()[0]].isna().sum()
            if NaCnt>0.1*self.resampleDF.shape[0]:
                raise RuntimeWarning('Bad choice of interval, which causes many NaNs.')
            # self.resampleDF.interpolate(inplace=True)

    def getOriginDataFrame(self, copy=False):
        if copy:
            return self.df.copy()
        return self.df

    def getResampleDataFrame(self, copy=False)->pd.DataFrame:
        if copy:
            return self.resampleDF.copy()
        return self.resampleDF

    def resetInterval(self, interval):
        self.resampleDF = self.df.resample(interval).mean()

    def loadConfig(self):
        with open(config, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def getColName(self):
        return self.config['ColName']

def read_csv(filePath)->pd.DataFrame:
    return pd.read_csv(filePath)
