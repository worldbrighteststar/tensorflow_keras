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
model.add(Dense(32, input_shape=(30,), activation='relu'))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid')) 

# 3. compile and train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

modelpath = './keras/checkPoint/k21_cancer_{epoch:02d}-{val_loss:.4f}.hdf5' # check point 저장 폴더/파일 포맷
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                             save_best_only=True, mode='auto') # save_best_only : 최고 성능 갱신시에만 저장

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=200, validation_split=0.2,
          callbacks=[early_stopping, checkpoint]
)

# 4. evaluate and predict
results = model.evaluate(x_test, y_test)
print(f'loss : {results[0]}')
print(f'acc : {results[1]}')

y_pred = model.predict(x_test)
print(f'actual y : {y_test}')
print(f'prediction : {y_pred}')

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