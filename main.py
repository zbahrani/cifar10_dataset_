import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.models import Sequential
from keras import datasets
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(train_image, train_label), (test_image, test_label) = datasets.cifar10.load_data()

# print(train_image.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# print(class_names[train_label[0][0]])

# Construction of convolution layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(32, 32, 3)))
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
              epochs=15, validation_data=(test_image, test_label))

Out = model.predict(test_image)

# print(Out[0])

m = -1000
o2 = Out[0]
ind = -1
for i in range(len(o2)):
    if o2[i] > m:
        m = o2[i]
        ind = i
print(class_names[ind])

plt.close()
plt.imshow(test_image[0])
plt.show()

# Enter a random photo for testing
img = cv2.imread('horse.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (32, 32))
# print(img.shape)
plt.close()
plt.imshow(img)
plt.show()
img = np.array([img])
out2 = model.predict(img)

m = -1000
o2 = out2[0]
ind = -1
for i in range(len(o2)):
    if o2[i] > m:
        m = o2[i]
        ind = i
print(class_names[ind])

