
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x = np.arange(1, 11) # = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,5,7,9,8,11])

# 2. model
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200)

# 4. evaluate and predict
y_pred = model.predict(x)
print(f'original y : {y}')
print(f'prediction : {y_pred}')

# 5. result visualization
import matplotlib.pyplot as plt

plt.scatter(x, y) # 기존 데이터 양상
plt.plot(x, y_pred, color='red') # 훈련 결과 weight
plt.show()