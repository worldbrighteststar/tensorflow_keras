from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()

# filters : filter의 개수, 해당 숫자 만큼의 feature map이 생긴다 -> 다음 layer의 input 개수
# kernel_size : filter의 크기
# strides : filter 이동 단위
# padding : image 둘레에 추가적인 pixcel 추가 -> feature map의 shape를 유지(or 변경) or 테두리 연산량 유지

model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, input_shape=(5,5,1)))
model.add(Conv2D(5, (2,2), padding='same')) # strides = 1(default)

model.add(Flatten()) # 2D data를 1D로 펼친다 -> Dense로 입력 가능
model.add(Dense(1, activation='sigmoid')) # ex. 이미지 데이터 이진 분류 

model.summary()