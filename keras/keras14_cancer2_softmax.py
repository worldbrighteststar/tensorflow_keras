import numpy as np

# 1. data
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target 
print(dataset.DESCR)
print(x.shape) # (569, 30)
print(y.shape) # (569,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.95, shuffle=True, random_state=66
)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_shape=(30,), activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2, activation='softmax')) # 이진 분류 모델 one-hot-encoding 필요 X(즉 output dim = 1), 활성 함수는 <sigmoid>

# 3. compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 이진 분류 모델 loss는 binary_crossentropy
model.fit(x_train, y_train, epochs=200, validation_split=0.2)

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss : {results[0]}')
print(f'acc : {results[1]}')

y_pred = model.predict(x_test)
print(f'actual y : {y_test}')
print(f'prediction : {y_pred}')