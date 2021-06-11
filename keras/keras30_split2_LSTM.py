import numpy as np
import tensorflow as tf

# 1. dataset
def split_x(seq, size): # sequance를 주소를 1씩 증가시키며 size만큼의 데이터로 나눈다. 
    aaa = []        
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset) 
    print(type(aaa))
    return np.array(aaa)

raw_data = np.array(range(1, 11))
print(f'base : {raw_data}\n')

size = 5
dataset = split_x(raw_data, size)
print(f'split_x : \n{dataset}\n')

x = dataset[:, :4]
y = dataset[:, -1]
print(f'x_data : \n{x}')
print(f'y_data : \n{y}')

x = x.reshape(6, 4, 1)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(4,1)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. complie and train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=1, epochs=200)

# 4. evaluate
results = model.evaluate(x, y)
print(f'loss : {results}')

x_pred = np.array([6,7,8,9])
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)
print(f'y_pred : {y_pred}')