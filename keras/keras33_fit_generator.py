import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. dataset
train_datagen = ImageDataGenerator(
    rescale=1./255 # 정규화
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(
    rescale=1./255, # 정규화
)

xy_train = train_datagen.flow_from_directory(
    directory='./tmp/horse-or-human',
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary' # y label format
)
xy_validation = validation_datagen.flow_from_directory(
    directory='./tmp/testdata',
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

print(xy_train[0][0].shape) # xdata (5, 300, 300, 3)
print(xy_train[0][1].shape) # ydata (5,)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(300,300,3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit_generator(xy_train, steps_per_epoch=8, epochs=15, validation_data=xy_validation)
model.fit(xy_train, steps_per_epoch=8, epochs=15, # 1027 trainSet, 128 batchSize => 8 steps
          validation_data=xy_validation, validation_steps=2) # 256 valSet, 128 batchSize => 2 valSteps

# 4. evaluate
# results = model.evaluate_generator(xy_validation)
results = model.evaluate(xy_validation)
print(results)
