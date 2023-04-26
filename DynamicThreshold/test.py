# test 2023/3/4 22:30
import numpy as np

from DataLayer import *
from Processor import *

df = pd.read_csv("data.csv")
dl = DataLayer(df, '60s')
data = dl.getResampleDataFrame()
train_data = data[:data.index[-10]]
print(train_data)
test_data = data[data.index[-10]:]
print(test_data)
processor = Processor(train_data)
ceemdan = processor.predict_with_CEEMDAN(10)
print(ceemdan)
normal = processor.predict_only_ARIMA(10)
print(normal)

