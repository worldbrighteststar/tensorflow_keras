import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#  model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=200)

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print(f'loss : {loss}')

result = model.predict([9])
print(f'prediction : {result}')