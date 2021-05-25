from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array

x = array(range(1, 101))
y = array(range(101, 201))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9
)

print(f'train set : {x_train.shape}')
print(f'test set  : {x_test.shape}')

# 2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(1))

# 3. complie and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size = 10, 
        validation_split=0.2) # sklearn으로 두번 split 할 필요 X

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(f'loss : {loss}')
y_pred = model.predict([101,102,103])
print(f'prediction : {y_pred}')