from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# 1. data
x = array(range(1, 101))
y = array(range(101, 201))

x_train, x_valid, x_test = x[:60], x[60:80], x[80:]
y_train, y_valid, y_test = y[:60], y[60:80], y[80:]

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size = 15, 
        validation_data=(x_valid, y_valid)) 

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(f'loss : {loss}')
y_pred = model.predict([101,102,103])
print(f'prediction : {y_pred}')