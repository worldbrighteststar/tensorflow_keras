import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113])

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=200)

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print(f'loss : {loss}')

results = model.predict(x_predict)
print(f'prediction : {results}')

"""
evaluate 자체는 모델 성능을 변화시키지 않는다.
"""