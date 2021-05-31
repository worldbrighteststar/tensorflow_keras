import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. data
from sklearn.datasets import load_boston
dataset = load_boston()

# sklearn datasets include (data, target, filename, feature_names, DESCR(description))
x = dataset.data # 특징들
y = dataset.target # 예측하고자 하는 목표값(label)

print(x.shape, y.shape) # (506, 13) (506,)
print(x[:5])
print(y[:5])
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9
)

# 2. model
input1 = Input(shape=(13,))
xx = Dense(128, activation='relu')(input1)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
output1 = Dense(1)(xx)
model = Model(inputs=input1, outputs=output1)
"""
model = Sequential()
model.add(Dense(128, input_shape=(13,), activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
"""

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, validation_split=0.2)

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
