from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# odd = []
# even = []
# for i in train_labels:
#     if i%2 == 0:
#         even.append(i)
#     else:
#         odd.append(i)
train_labels_evenodd = train_labels % 2
test_labels_evenodd = test_labels % 2

train_images_4d = train_images.reshape((60000,28,28,1))
test_images_4d = test_images.reshape((10000,28,28,1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input

input_images = Input(shape=(28,28,1), dtype='int32', name='images')
x = layers.Conv2D(filters=64, kernel_size = (3,3), activation= 'relu')(input_images)
x = layers.Conv2D(filters=128, kernel_size=(3,3), activation= 'relu')(x)
x = layers.MaxPool2D(pool_size=3)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dropout(0.4)(x)

# images
pred_images = layers.Dense(10, activation = 'softmax', name = 'number')(x)

# evenodd
pred_evenodd = layers.Dense(1,activation = 'sigmoid', name = 'evenodd')(x)

model = Model(images_input, [pred_images, pred_evenodd])

model.compile(optimizer = 'adam',
             loss = {'number':'sparse_categorical_crossentropy',
                     'evenodd': 'binary_crossentropy'},
             metrics = ['acc'])

import tensorflow as tf