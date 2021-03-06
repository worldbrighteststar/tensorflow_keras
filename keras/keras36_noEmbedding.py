from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밌네요', '선생님 잘 생기긴 했어요']

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1]) # 긍정 : 1, 부정 : 0

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, 
# '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '선생님': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x) 
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], 
# [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x)
print(pad_x.shape) # (13, 5)
print(len(np.unique(pad_x))) # 28

pad_x = pad_x.reshape(13, 5, 1) # Embedding 하지 않고 바로 LSTM으로 보내기. 가능하다는 것만 알아두자.

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()

model.add(LSTM(32, activation='relu', input_shape=(5,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. compile and train
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

# 4. predict
y_pred = model.predict([[[0], [0], [0], [25], [19]]]) # [0, 0, 0, 25, 19] means "선생님 지루해요"
print(y_pred)