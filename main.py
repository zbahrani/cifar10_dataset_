import tensorflow as tf

from keras import datasets
from keras.layers import Dense

(train_image, train_label), (test_image, test_label) = datasets.cifar10.load_data()
print(train_image.shape)