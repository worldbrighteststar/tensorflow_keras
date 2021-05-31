import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([range(100), range(301, 401), range(1, 101), range(101, 201), range(401, 501)])
y = np.array([range(1, 101), range(201, 301)])
x = np.transpose(x)
y = np.transpose(y)
print(x.shape) # (100, 5)
print(y.shape) # (100, 2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(20))
model.add(Dense(2))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=20)

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(f'loss(mse) : {loss}')
y_pred = model.predict(x_test)
print(f'prediction : {y_pred}')

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # prams : (실제값, 예측값)
    return np.sqrt(mean_squared_error(y_test, y_predict))

print(f'MSE  : {mean_squared_error(y_test, y_pred)}') # = loss
print(f'RMSE : {RMSE(y_test, y_pred)}')

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print(f'R2   : {R2}')
