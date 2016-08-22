from __future__ import print_function
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import \
 Convolution2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from prepare_data import load_data, load_train, predict
import cPickle as pickle
from dataset import *


class Model(Sequential):
    def __init__(self, model_name, nb_classes, nb_epoch,
                 batch_size, img_row, img_column, img_channel):
        super(Model, self).__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.img_row = img_row
        self.img_column = img_column
        self.img_channel = img_channel
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.prepared = False
        self.trained = False

        # Convolution layers
        self.add(ZeroPadding2D(input_shape=(3, 32, 32), padding=(2, 2)))
        self.add(Convolution2D(
            64, 5, 5, border_mode='valid', activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), border_mode='same'))

        self.add(ZeroPadding2D(padding=(2, 2)))
        self.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), border_mode='same'))

        self.add(ZeroPadding2D(padding=(2, 2)))
        self.add(Convolution2D(128, 5, 5, border_mode='valid', activation='relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), border_mode='same'))

        # Fully connected layer
        self.add(Flatten())
        self.add(Dense(1000))
        self.add(Dropout(0.25))
        self.add(Activation('relu'))
        self.add(Dense(self.nb_classes))
        self.add(Activation('softmax'))

        self.optimizer = Adam(lr=0.001, epsilon=1e-8)

    def set_train_data(self, x, y):
        """input x is 2d matrix. y is vector."""
        # self.x_train = x.reshape(
            # y.shape[0], self.img_channel, self.img_row, self.img_column
            # ).astype(np.float32) / 255
        self.y_train = np_utils.to_categorical(y)
        self.x_train = x
        # self.y_train = npy
        print('Successfully set train data. Train model.')
        self.prepared = True

    def save(self):
        super(Model, self).save(os.path.join('data/models/', self.model_name))

    def train(self, validation_data=None):
        if self.prepared:
            self.compile(loss='categorical_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])
            self.fit(self.x_train, self.y_train,
                     batch_size=self.batch_size,
                     nb_epoch=self.nb_epoch,
                     #  validation_split=0.1,
                     validation_data=validation_data,
                     shuffle=True,
                     show_accuracy=True)

        else:
            print('Please set data to train model.')


if __name__ == '__main__':
    with open('data/image_norm_zca.pkl', 'rb') as f:
        images = pickle.load(f)
        index = np.random.permutation(len(images['train']))
        train_index = index[:-5000]
        valid_index = index[-5000:]
        train_x = images['train'][train_index].reshape((-1, 3, 32, 32))
        valid_x = images['train'][valid_index].reshape((-1, 3, 32, 32))
        test_x = images['test'].reshape((-1, 3, 32, 32))

    with open('data/label.pkl', 'rb') as f:
        labels = pickle.load(f)
        train_y = labels['train'][train_index]
        valid_y = labels['train'][valid_index]
        valid_y = np_utils.to_categorical(valid_y)
        test_y = labels['test']


    model = Model('cifar_with_norm', 10, 25, 124, 32, 32, 3)
    model.set_train_data(train_x, train_y)
    model.train(validation_data=(valid_x, valid_y))
    model.save()
    predict(test_x, test_y)
