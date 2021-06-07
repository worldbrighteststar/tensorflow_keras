import numpy as np

# 1. dataset
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

from tensorflow.keras.utils import to_categorical # one-hot encoding & categorical_crossentropy
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# 1.5 normalization : min-max scaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=3, strides=1,
                padding='same', input_shape=(32,32,3), activation='relu')) 
model.add(Conv2D(30, 3, activation='relu'))
model.add(Conv2D(30, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=30, validation_split=0.2)

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss : {results[0]}')
print(f'acc  : {results[1]}')
