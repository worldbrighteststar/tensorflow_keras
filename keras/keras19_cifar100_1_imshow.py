import numpy as np
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) # (50000, 1) (10000, 1)

"""
shape가 cifar-10과 같기 때문에 'class 분류 개수'만 고려하면 모델이 같아도 된다.
cifar-10 : 10 classes -> Dense(10, 'softmax')
cifar-100 : 100 classes -> Dense(100, 'softmax')
"""

print(x_train[0])
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[0]) # x_train[0] : 소!
plt.show()
