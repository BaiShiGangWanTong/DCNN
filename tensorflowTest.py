import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras as kr
import torch


print('Tensorflow version: {}'.format(tf.__version__))

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class SCNN(kr.Model):
    def __init__(self):
        super(SCNN, self).__init__()
        self.c1 = kr.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))
        self.pooling1 = kr.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.c2 = kr.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))
        self.pooling2 = kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.c3 = kr.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=(1, 1))
        self.pooling3 = kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = kr.layers.Dense(2048, activation='softmax')
        self.d = tf.keras.layers.Dropout(0.5)
        self.fc2 = kr.layers.Dense(2, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)  # L1
        x = tf.nn.relu(x)
        x = self.c1(x)  # L2
        x = tf.nn.relu(x)
        x = self.pooling1(x) # L3

        x = self.c2(x)  # L4
        x = tf.nn.batch_normalization(x)
        x = tf.nn.relu(x)

        x = self.c2(x)  # L5
        x = tf.nn.batch_normalization(x)
        x = tf.nn.relu(x)
        x = self.pooling2(x)  # L6

        x = self.c3(x)  # L7
        x = tf.nn.batch_normalization(x)
        x = tf.nn.relu(x)

        x = self.c3(x)  # L8
        x = tf.nn.batch_normalization(x)
        x = tf.nn.relu(x)

        x = self.pooling3(x)  # L9
        x = self.flatten(x)
        x = self.fc1(x)  # L10
        x = self.d(x)
        x = self.fc1(x)  # L11
        x = self.d(x)
        x = self.fc2(x)  # L12
        return x


model = SCNN()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

# show
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print(acc)
print(val_loss)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()




