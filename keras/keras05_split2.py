from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array

# 1. data
x = array(range(1, 101))
y = array(range(101, 201))

# manually split : Data will be sequencially splited
# x_train, x_valid, x_test = x[:60], x[60:80], x[80:]
# y_train, y_valid, y_test = y[:60], y[60:80], y[80:]

# randomly shuffle split using sklearn : randomly splited 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( # train, test split
    x, y, train_size=0.9, shuffle=True # shuffle : default is True
)
# print(x_train) # shuffled
# print(x_test) # also shuffled
# print(x_train.shape)
# print(x_test.shape)

x_train, x_valid, y_train, y_valid = train_test_split( # train, valid split
    x_train, y_train, train_size=0.8
) 
print(f'train set : {x_train.shape}')
print(f'valid set : {x_valid.shape}')
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
model.fit(x_train, y_train, epochs=200, batch_size = 15, 
        validation_data=(x_valid, y_valid)) 

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(f'loss : {loss}')
y_pred = model.predict([101,102,103])
print(f'prediction : {y_pred}')

"""
shuffle 해주는 이유 : 순차적으로 data를 자르면 train되는 data 범주와 test에 사용되는 data 범주가
구분되기 때문에 실제 prediction에서의 성능이 떨어질 수 있다.
shuffle하게 되면 train과 test로 분리되는 비율은 같지만 랜덤하게 분리되기 때문에 각 set의 데이터가 
거의 비슷하게 분포하게 된다. 
"""