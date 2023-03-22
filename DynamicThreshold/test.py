# test 2023/3/4 22:30
import numpy as np

from DataLayer import *
from Processor import *

df = read_csv('data2.csv')
dl = DataLayer(df, '60s')
data = dl.getResampleDataFrame()
print(data.head(10))
# processor = Processor(data)
# processor.predict(20)








