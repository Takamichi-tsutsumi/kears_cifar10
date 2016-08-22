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
                     validation_split=0.1,
                     #  validation_data=validation_data,
                     shuffle=True,
                     show_accuracy=True)

        else:
            print('Please set data to train model.')


if __name__ == '__main__':
    x_train, y_train = load_train()

    x_test, y_test_classes = load_data('test_batch')
    model = Model('cifar_cnn_with_momentum', 10, 25, 124, 32, 32, 3)
    model.set_train_data(x_train, y_train)
    model.train()
    predict(model, x_test, y_test_classes)
    model.save()
