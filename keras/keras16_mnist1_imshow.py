import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000,) (10000,)
 
print(x_train[0])
print(y_train[0])

# 시각화
import matplotlib.pyplot as plt

plt.imshow(x_train[0]) # x_train[0] : 5
plt.show()