import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([[100,85,70],[90,85,100],
              [80,50,30],[43,60,100]])
y = np.array([75,65,33,85])

# 2. model
model = Sequential()
model.add(Dense(100, input_shape=(3,)))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

# 3. compile and train
model.compile(loss='mse',
              optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

# 4. evaluate
loss = model.evaluate(x, y)
print(f'loss : {loss}')