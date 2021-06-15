from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])
# word_index 번호 순서 : order by 출연 빈도, 출연 순서
print(token.word_index) # {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

x = token.texts_to_sequences([text])
print(x) # [[3, 1, 1, 4, 5, 2, 2, 6]]

# word_index가 대소 관계로 적용되지 않도록 one-hot encoding을 해보자.
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index) # 6
x = to_categorical(x)
print(x)
print(x.shape) # (1, 8, 7) 
"""
왜 8,6이 아닐까?
to_categorical은 무조건 시작 번호가 0이다.
x는 [1,2,3,4,5,6]의 6개의 class로 분류되지만 
0이 포함되어 one-hot encoding 크기는 7이다.  
"""