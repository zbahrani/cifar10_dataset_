import tensorflow as tf
import matplotlib.pyplot as plt


from keras import datasets
from keras.layers import Dense

(train_image, train_label), (test_image, test_label) = datasets.cifar10.load_data()

print(train_image.shape)

plt.imshow(train_image[0])
plt.show()