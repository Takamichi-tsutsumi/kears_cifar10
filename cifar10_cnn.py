from __future__ import print_function
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from prepare_data import load_data, load_train, predict
import cPickle as pickle


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

        # CNN model below
        self.add(Convolution2D(6, 3, 3, border_mode='valid', input_shape=(3, 32, 32)))
        self.add(Activation('relu'))
        self.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
        self.add(Convolution2D(12, 3, 3, border_mode='valid'))
        self.add(Activation('relu'))
        self.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
        self.add(Flatten())
        self.add(Dense(120))
        self.add(Dropout(0.5))
        self.add(Activation('relu'))
        self.add(Dense(84))
        self.add(Dropout(0.5))
        self.add(Activation('relu'))
        self.add(Dense(nb_classes))
        self.add(Activation('softmax'))

        self.optimizer = SGD(lr=0.01)

    def set_train_data(self, x, y):
        """input x is 2d matrix. y is vector."""
        self.x_train = x.reshape(
            y.shape[0], self.img_channel, self.img_row, self.img_column
            ).astype(np.float32) / 255
        self.y_train = np_utils.to_categorical(y)
        print('Successfully set train data. Train model.')
        self.prepared = True

    def save(self):
        super(os.path.join('data/models/', self.model_name))

    def predict(self, x_test, y_test):
        predict_classes = self.predict_classes(x_test)
        accuracy = [x == y for (x, y) in zip(predict_classes, y_test)]
        print(accuracy)
        acc_rate = sum(i for i in accuracy if i) / float(len(y_test)) * 100
        print('accuracy:{}'.format(acc_rate))

    def train(self, validation_data=None):
        if self.prepared:
            self.compile(loss='categorical_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])
            self.fit(self.x_train, self.y_train,
                     batch_size=self.batch_size,
                     nb_epoch=self.nb_epoch,
                     validation_data=validation_data,
                     shuffle=True)

        else:
            print('Please set data to train model.')



if __name__ == '__main__':
    x_train, y_train = load_train()

    x_test, y_test_classes = load_data('test_batch')
    model = Model('cifar_cnn', 10, 50, 32, 32, 32, 3)
    model.set_train_data(x_train, y_train)
    model.train()
    model.predict(x_test, y_test_classes)
    model.save()
