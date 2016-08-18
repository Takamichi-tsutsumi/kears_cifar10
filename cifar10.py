import os
import cPickle as pickle
import numpy as np
from prepare_data import load_data, load_train


x_train, y_train = load_train()
x_test, y_test = load_data('test_batch')
