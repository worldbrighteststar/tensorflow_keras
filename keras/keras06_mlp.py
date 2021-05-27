import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) # (2, 10) : 2행 10열
print(y.shape) # (10,) : 1벡터(차원) 10스칼라 = 10행 1열
# 따라서 x, y 매칭 불가 -> x를 10행 2열로 변경해야 한다.
# np.transpose(x) : 행열 변환
x = np.transpose(x) # x = [[1,11],[2,12],[3,13], ...]
print(x.shape) # (10, 2)
print(y.shape) # (10,)

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
y_pred = model.predict(np.transpose([[11,12,13],[21,22,23]])) # actual input = [[11,21],[12,22],[13,23]]
print(f'prediction : {y_pred}')

"""
np.array().shape은 '행과 열'
model에서 받는 input_shape는 데이터 '하나'의 모양이다.
"""