import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

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
model = Sequential()
model.add(Dense(128, input_shape=(30,), activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2, activation='softmax')) # 이진 분류는 다중 분류에 포함되므로 다중 분류와 같은 setting 가능

# 3. compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # one-hot encoding(manually) + categorical_crossentropy 와 같다.

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto') # monitor가 mode 방향과 반대로 patience만큼 가면 stop
                                                                # mode = if loss : min or auto if acc : max or auto
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, 
                callbacks=[early_stopping]) # 훈련에 적용하는 callbacks 관련 인자  

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss : {results[0]}')
print(f'acc : {results[1]}')

y_pred = model.predict(x_test)
print(f'actual y : {y_test}')
print(f'prediction : {y_pred}')

print(hist) # <tensorflow.python.keras.callbacks.History object xx>
print(hist.history.keys()) # dict_keys(['loss', 'acc', 'val_loss', 'val_acc']) 
# print(hist.history['loss'])
print(hist.history['val_loss'])

# 5. visualization
import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.title('train_loss & val_loss')
# plt.xlabel('epochs') 
# plt.ylabel('loss')
# plt.legend(['train loss', 'val_loss'])

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('train_acc & val_acc')
plt.xlabel('epochs') 
plt.ylabel('acc')
plt.legend(['train acc', 'val_acc'])

plt.show()