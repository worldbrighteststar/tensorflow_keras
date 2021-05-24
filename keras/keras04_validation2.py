import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. dataset to train and validation set (and test set)
x_train = np.array([1,2,3,4,5,6,7,14,15,16])
y_train = np.array([1,2,3,4,5,6,7,14,15,16])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(20))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1, 
        #  validation_data=(x_valid, y_valid))
        validation_split=0.3) # trainset의 30%를 validation set으로 사용

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print(f'loss : {loss}')
result = model.predict([9])
print(f'prediction : {result}')