import numpy as np

# 1. dataset
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)

# 1.5 normalization : min-max scaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

model = Sequential()

# model.add(LSTM(32, activation='relu', input_shape=(28,28)))

model.add(Conv1D(filters=30, kernel_size=2, strides=1,
                 padding='same', input_shape=(28,28))) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss : {results[0]}')
print(f'acc  : {results[1]}')
