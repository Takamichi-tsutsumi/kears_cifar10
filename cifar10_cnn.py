from __future__ import print_function
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from prepare_data import load_data, load_train, predict

batch_size = 32
nb_classes = 10
nb_epoch = 100
img_row = 32
img_column = 32
img_channel = 3

x_train, y_train = load_train()
x_train = x_train.reshape(y_train.shape[0], 3, 32, 32).astype(np.float32) / 255
y_train = np_utils.to_categorical(y_train)

x_test, y_test_classes = load_data('test_batch')
x_test = x_test.reshape(y_test_classes.shape[0], 3, 32, 32).astype(np.float32) / 255
y_test = np_utils.to_categorical(y_test_classes)

print('datashape train {}'.format(x_train.shape))
print('datashape test {}'.format(x_test.shape))

model = Sequential()
model.add(Convolution2D(6, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(Convolution2D(12, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

optimizer = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save('./data/models/cnn_model.h5')

predict(model, x_test, y_test_classes)
