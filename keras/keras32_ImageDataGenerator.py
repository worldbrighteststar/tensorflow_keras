import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 전처리, 증강 명시
train_datagen = ImageDataGenerator(
    rescale=1./255, # 정규화
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(
    rescale=1./255, # 정규화
)

# ImageDataGenerator 내용을 데이터에 적용
xy_train = train_datagen.flow_from_directory(
    directory='./tmp/horse-or-human',
    target_size=(300, 300),
    batch_size=5,
    class_mode='binary' # y label format
) # 1027 images belonging to 2 classes.
xy_validation = validation_datagen.flow_from_directory(
    directory='./tmp/testdata',
    target_size=(300,300),
    batch_size=5,
    class_mode='binary'
) # 256 images belonging to 2 classes.

print(xy_train[0][0].shape) # xdata (5, 300, 300, 3)
print(xy_train[0][1].shape) # ydata (5,)