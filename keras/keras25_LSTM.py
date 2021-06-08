import numpy as np

# 1. dataset
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1)
print(x.shape) # (4, 3, 1)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

"""
Dense : (N, col)
Conv2D : (N, width, height, channel)
LSTM : (N, col, 몇개씩 자르는지)
"""
model = Sequential()
model.add(LSTM(10, input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(1))

model.summary()