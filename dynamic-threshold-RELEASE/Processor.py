# CIMFS 2023/3/20 19:29
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from matplotlib import pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_squared_error


class Processor:
    def __init__(self, data: pd.DataFrame):
        """
        接受具有时间戳的等间隔数据
        :param data:
        """
        self.data = data.copy()

    def predict_only_ARIMA(self, steps):
        # 无CEEMDAN，纯ARIMA，可作为对照组
        def forward_index(begin):
            interval = self.data.index[1] - self.data.index[0]
            datetime = self.data.index[begin] + interval
            for i in range(steps):
                yield datetime
                datetime += interval

        df = pd.DataFrame()
        # 整体上是对每一列进行操作的
        for col in self.data:
            series = self.data[col]
            model = pm.auto_arima(series,
                                  stepwise=False,
                                  n_jobs=-1)
            predict = model.predict(steps)
            ans = pd.Series(predict, index=forward_index(-1))
            df[col] = ans
        return df

    def predict_with_CEEMDAN(self, steps: int) -> pd.DataFrame:
        """
        :param steps: 向后预测的步长个数
        :return: 返回具有时间戳索引的预测数据
        """
        def forward_index(begin):
            interval = self.data.index[1] - self.data.index[0]
            datetime = self.data.index[begin] + interval
            for i in range(steps):
                yield datetime
                datetime += interval

        df = pd.DataFrame()
        # 整体上是对每一列进行操作的
        for col in self.data:
            # 第一步，进行ceemdan分解
            series = self.data[col]
            cIMFs, res = get_cIMFs(series)
            print(col)
            if np.isnan(cIMFs).any():
                df[col] = pd.Series(np.full(steps, series[-1]), index=forward_index(-1))
                continue
            total = np.zeros(steps)
            for cimf in cIMFs:
                try:
                    model = pm.auto_arima(cimf,
                                          stepwise=False,
                                          n_jobs=-1)
                    predict = model.predict(steps)
                except Exception:
                    predict = np.full(steps, cimf[-1])
                total += predict
            ans = pd.Series(total, index=forward_index(-1))
            df[col] = ans
        return df


def get_cIMFs(series: pd.Series):
    """
    本函数是原始的版本，返回的数据是np.ndarray
    :param series: pandas的Series类型数据，具有时间戳索引
    :return: cIMFs: , res: pd.Series
    """
    seq = series.values
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(seq)
    cIMFs, res = ceemdan.get_imfs_and_residue()
    return cIMFs, res


def get_indexed_cIMFs(series: pd.Series):
    """
    本函数对ceemdan的函数进行了包装，保证输入和输出都是基于pd.Series的数据，并维护了索引不变
    :param series: pandas的Series类型数据，具有时间戳索引
    :return: cIMFs: list[pd.Series], res: pd.Series
    """
    if type(series) != pd.Series:
        raise TypeError('Expected pd.Series,', type(series), 'however.')
    seq = series.values
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(seq)
    cIMFs, res = ceemdan.get_imfs_and_residue()
    cIMFs = [pd.Series(cimf, index=series.index) for cimf in cIMFs]
    res = pd.Series(res, index=series.index)
    return cIMFs, res


def show(series, cIMFs, res):
    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(hspace=0.1)
    plt.subplot(cIMFs.shape[0] + 2, 1, 1)
    plt.plot(series, 'r')
    plt.ylabel('OriginData')
    for i in range(cIMFs.shape[0]):
        plt.subplot(cIMFs.shape[0] + 3, 1, i + 3)
        plt.plot(cIMFs[i], 'g')
        plt.ylabel("IMF %i" % (i + 1))
        plt.locator_params(axis='x', nbins=10)
    plt.subplot(cIMFs.shape[0] + 3, 1, cIMFs.shape[0] + 3)
    plt.plot(res, 'g')
    plt.ylabel('Residual')
    plt.show()
