import numpy as np

# 1. dataset
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1) 
print(x_train.shape) # (60000, 28, 28, 1)

from tensorflow.keras.utils import to_categorical 
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# 1.5 normalization : min-max scaler
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2), strides=1,
                padding='same', input_shape=(28,28,1), activation='relu')) 
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('./keras/model/k23_1_model_1.h5') # compile 이전에 저장

# 3. compile and train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

model.save('./keras/model/k23_1_model_2.h5') # compile 이후 train 이전에 저장

model.fit(x_train, y_train, epochs=10, validation_split=0.2)

model.save('./keras/model/k23_1_model_3.h5') # train 이후에 저장