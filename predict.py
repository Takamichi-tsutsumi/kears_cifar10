import numpy as np
from keras.models import load_model
from prepare_data import load_data, predict

if __name__ == '__main__':

    x_test, y_test_classes = load_data('test_batch')
    x_test = x_test.reshape(y_test_classes.shape[0], 3, 32, 32).astype(np.float32) / 255

    model = load_model('./data/models/cnn_model.h5')
    predict(model, x_test, y_test_classes)
