import numpy as np

# 1. dataset
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
              [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], 
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
print(x.shape) # (13, 3)
print(y.shape) # (13,)

x = x.reshape(13, 3, 1)
print(x.shape) # (13, 3, 1)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

"""
Dense, Conv2D는 input과 output이 동일한 차원을 가지기 때문에
연속해서 쌓을 수 있다. 
LSTM은 input이 3차원, output은 2차원이다.
LSTM을 이어서 쌓기 위해 return_sequences=True 로 설정.
"""
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. complie and train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=1, epochs=200)

# 4. evaluate
results = model.evaluate(x, y)
print(f'loss : {results}')

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1,3,1)
y_pred = model.predict(x_pred)
print(f'y_pred : {y_pred}')
