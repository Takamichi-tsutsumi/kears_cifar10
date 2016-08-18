from __future__ import print_function
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from prepare_data import load_data, load_train

batch_size = 32
nb_classes = 10
nb_epoch = 100
img_size = 32 * 32
channel_size = 3

x_train, y_train = load_train()
y_train = np_utils.to_categorical(y_train)
x_test, y_test = load_data('test_batch')
y_test = np_utils.to_categorical(y_test)

print('datashape train {}'.format(x_train.shape))
print('datashape test {}'.format(x_test.shape))

model = Sequential()
model.add(Dense(1024, input_dim=img_size * channel_size, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='relu'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')


model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
loss_and_metrics = model.evaluate(x_test, y_test)
print(loss_and_metrics)
