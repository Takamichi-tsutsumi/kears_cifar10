import os
import cPickle as pickle
import numpy as np


def load_data(filename, as_list=False):
    file_path = os.path.join('./data/cifar-10-batches-py/', filename)
    f = open(file_path, 'r')
    dic = pickle.load(f)
    f.close()
    data = dic['data']
    labels = dic['labels']
    if as_list:
        return data, labels
    else:
        return np.array(data), np.array(labels)


def load_train():
    filenames = ['data_batch_{}'.format(n) for n in range(1, 6)]
    data = []
    labels = []
    for f in filenames:
        d, l = load_data(f, as_list=True)
        data.extend(d)
        labels.extend(l)

    return np.array(data), np.array(labels)


x_train, y_train = load_train()
x_test, y_test = load_data('test_batch')
