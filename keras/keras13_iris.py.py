import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. data
from sklearn.datasets import load_iris
dataset = load_iris()

# sklearn datasets include (data, target, filename, feature_names, DESCR(description))
x = dataset.data # 특징들
y = dataset.target # 예측하고자 하는 목표값(label)

print(x.shape, y.shape) # (150, 4) (150,)
print(x[:5])
print(y[:5])
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=66
)

# 2. model
input1 = Input(shape=(4,))
xx = Dense(128, activation='relu')(input1)
xx = Dense(64)(xx)
xx = Dense(32)(xx)
xx = Dense(16)(xx)
output1 = Dense(3, activation='softmax')(xx) # 다중 분류 모델 output layer의 node는 <label 개수>, 활성 함수는 <softmax>
model = Model(inputs=input1, outputs=output1)
"""
model = Sequential()
model.add(Dense(128, input_shape=(4,), activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))
"""

# 3. complie and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중 분류 모델 loss는 crossentropy 계열, ex:sparse_categorical_crossentropy
model.fit(x_train, y_train, epochs=200, verbose=1, validation_split=0.2)

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss(mse) : {results[0]}')
print(f'acc : {results[1]}')

y_pred = model.predict(x_test)
print(y_test)
print(f'prediction : {y_pred}')
