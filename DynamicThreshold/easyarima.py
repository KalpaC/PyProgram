# easy_arima 2023/3/20 19:36
from copy import copy

from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace import sarimax

from sklearn.model_selection import GridSearchCV

from statsmodels.tsa.stattools import adfuller

import pandas as pd

import pmdarima as pm


class EasyArima:
    def __init__(self, data: pd.Series, forward: int):
        self.data = data
        self.diffs = pd.DataFrame(self.data, columns=[0])
        self.d = 0
        self.stabilize()
        if not self.is_correlated():
            # 如果是白噪声，可以考虑直接用均值或者最后一个值作为预测值。
            pass
        # 确定了差分阶数d，获得了平稳的数据，然后需要搜索p,q
        p,q = evaluate_models(self.diffs[self.d],p_values=range(0,6),q_values=range(0,6))


    def is_stable(self):
        ans = adfuller(self.diffs[self.d], autolag='AIC')
        return ans[1] < 0.05

    def stabilize(self):
        while not self.is_stable():
            self.d += 1
            self.diffs[self.d] = self.diffs[self.d - 1].diff()

    def is_correlated(self):
        lbvalue, pvalue = acorr_ljungbox(self.diffs[self.d], lags=20)
        print(pvalue)
        return min(pvalue) < 0.05


# 尝试采用均方误差作为评估标准。

def evaluate_arima_model(data, p, q):
    # 数据已经完成了差分，不再设置d
    train_size = int(len(data) * 0.66)
    train, test = data[:train_size], data[train_size:]
    # 由于ARIMA模型一次只能预测一个值，所以需要滚动预测
    history = [x for x in train]  # 滚动数组
    predict = []  # 预测数组
    for i in range(len(test)):
        model = ARIMA(history, order=(p, 0, q))
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predict.append(yhat)
        history.append(test[i])
    error = mean_squared_error(test, predict)
    return error


def evaluate_models(dataset, p_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_p, best_q = float("inf"), 0, 0
    for p in p_values:
        for q in q_values:
            try:
                mse = evaluate_arima_model(dataset, p, q)
                if mse < best_score:
                    best_score, best_p, best_q = mse, p, q
                print('ARIMA(%d,%d) MSE=%.3f' % (p, q, mse))
            except:
                continue
    print('Best ARIMA(%d,%d) MSE=%.3f' % (best_p, best_q, best_score))
    return best_p, best_q
