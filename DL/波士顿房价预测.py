# 作者 Ajex
# 创建时间 2023/4/3 20:40
# 文件名 波士顿房价预测.py
from keras.datasets import boston_housing
from keras import models, layers
import numpy as np

# 加载数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据处理：标准化。由于每个样本的模相同（和影评不同），所以无需向量化。但是数量的单位不统一
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# 构建网络
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# K折交叉验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #%d' % i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]], axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]], axis=0
    )

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
