import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)
"""
shape가 mnist와 같기 때문에 모델이 같아도 된다.
"""
print(x_train[0])
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[0]) # x_train[0] : Ankle boot
plt.show()