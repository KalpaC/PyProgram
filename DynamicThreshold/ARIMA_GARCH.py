# ARIMA_GARCH 2023/1/5 16:04
# ARIMA模型
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
# 稳定性检验
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np


class MyARIMA:
    def __init__(self, data: pd.Series):
        self.data = data
        self.afterDiff = self.data
        self.d = 0

    def predict_forecast(self, predictFrom, predictTo):
        """
        :param predictFrom: 预测数据的起始下标
        :param predictTo: 预测数据的结束下标
        :return:(predict_data, forecast_data)
        """
        self.stabilize()
        if not self.isCorrelated():
            print('is not correlated')
            return None

        p, q = self.getMinOrder()

        print(
            "最终定阶\np:", p, '\n',
            'd:', self.d, '\n',
            "q:", q
        )
        # 建立最终模型
        result = self.get_model_result(p, q)

        print()
        print(result.summary())

        # 利用模型进行预测,predict
        fitting = result.predict(start=predictFrom, end=predictTo)
        # 展示拟合值与原始值的区别
        # plt.plot(self.afterDiff, 'r')
        # plt.plot(fitting, 'b')
        # plt.show()

        real = self.diffReduction(fitting)
        plt.plot(self.data, 'r')
        plt.plot(real, 'b')
        plt.show()

        # return predict, forecast

    def diffReduction(self, diff: pd.DataFrame):
        """
        :param diff: 差分序列
        :return:
        """
        last = diff.index[0] - 1
        beforeLast = diff.index[0] - 2

        if self.d == 0:
            return diff
        if self.d == 1:
            real = diff.cumsum() + self.data[last]
            return real.dropna()
        if self.d == 2:
            diff1 = diff.cumsum() + (self.data[last] - self.data[beforeLast])
            real = diff1.cumsum() + self.data[beforeLast]
            return real.dropna()
        raise Exception("过高的差分次数")

    def stabilize(self):
        while not self.isStable():
            self.afterDiff = self.afterDiff.diffs()[1:]
            print(self.afterDiff.head(), end='\n\n')
            self.d += 1

    def evaluate(self, predict, start, end):
        mse = mean_squared_error(self.afterDiff[start:end + 1], predict, multioutput='uniform_average')
        return mse

    def isCorrelated(self):
        lbvalue, pvalue = acorr_ljungbox(self.afterDiff, lags=20)
        print(pvalue)
        return min(pvalue) < 0.05

    def isStable(self):
        ans = adfuller(self.afterDiff, autolag='AIC')
        print('pvalue:', ans[1])
        return ans[1] < 0.05

    def getMinOrder(self):
        # analysis = arma_order_select_ic(self.afterDiff, max_ar=5, max_ma=5, ic=['bic'])
        # print(analysis)
        # return analysis['bic_min_order']
        print(self.data)
        # 尝试使用网格搜索
        pmax = 5
        qmax = 5
        aic_matrix = []
        for p in range(pmax + 1):
            temp = []
            for q in range(qmax + 1):
                try:
                    temp.append(ARIMA(self.data, order=(p, self.d, q)).fit().aic)
                except:
                    temp.append(np.nan)
            aic_matrix.append(temp)
        print(aic_matrix)
        aic_matrix = pd.DataFrame(aic_matrix)
        p, q = aic_matrix.stack().idxmin()
        return p, q

    def get_model_result(self, p, q):
        model = ARIMA(self.data, (p, self.d, q))
        return model.fit()
