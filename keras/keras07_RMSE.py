import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.transpose(x)

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu')) # input_shape=(2,)
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size = 1)

# 4. evaluate and predict
loss = model.evaluate(x, y)
print(f'loss(mse) : {loss}')
y_pred = model.predict(x)
print(f'prediction : {y_pred}')

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # prams : (실제값, 예측값)
    return np.sqrt(mean_squared_error(y_test, y_predict))

print(f'MSE  : {mean_squared_error(y, y_pred)}')
print(f'RMSE : {RMSE(y, y_pred)}')