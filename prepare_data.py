import os
from os import path
from urllib import urlretrieve
import tarfile

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filename = url.split('/')[-1]
filepath = path.join('./data', filename)

if not path.exists(filepath):
    urlretrieve(url, path.join('./data', filename))
    tar = tarfile.open(filepath, 'r:gz')
    tar.extractall('./data')
    tar.close()
