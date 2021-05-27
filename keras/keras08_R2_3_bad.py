# R2를 음수가 아닌 0.5 이하로 만들기
# 1. layer 6개 이상
# 2. batch_size=1
# 3. epochs=100 이상
# 4. Hidden layer 노드 범위 (10, 1000)
# 5. 데이터 조작 X

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='softmax'))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

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