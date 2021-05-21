import numpy as np
import tensorflow as tf

# 1. '정제된' data
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() # 모델의 큰 틀을 구성 : Sequential 모델
model.add(Dense(3, input_dim=1))
model.add(Dense(4)) # Sequential 모델이기 때문에 input_dim은 이전 layer의 node개수
model.add(Dense(2))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam') # 모델의 loss function, optimizer를 설정
model.fit(x, y, batch_size=1, epochs=500)

# 4. evaluate and predict
loss = model.evaluate(x, y, batch_size=1)
print(f'loss : {loss}')

result = model.predict([4])
print(f'prediction : {result}')