import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import datasets
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(train_image, train_label), (test_image, test_label) = datasets.cifar10.load_data()

# print(train_image.shape)

# plt.imshow(train_image[0])
# plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# print(class_names[train_label[0][0]])

model = Sequential()
model.add(Conv2D(32 , kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
h = model.fit(train_image, train_label,
              epochs=10, validation_data=(test_image, test_label))
