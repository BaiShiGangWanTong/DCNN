import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import keras.layers as kl
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# 数据所在文件夹
base_dir = './dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 训练集
train_fire_dir = os.path.join(train_dir, 'fire')
train_no_fire_dir = os.path.join(train_dir, 'nofire')

# 验证集
val_fire_dir = os.path.join(val_dir, 'fire')
val_nofire_dir = os.path.join(val_dir, 'nofire')

# model = tf.keras.models.Sequential([
#     kl.Conv2D(kernel_size=(3, 3), filters=32, strides=(1, 1), padding='same', activation='relu', input_shape=(224, 224, 3)),  # L1
#     kl.Conv2D(kernel_size=(3, 3), filters=32, strides=(1, 1), padding='same', activation='relu'),  # L2
#     kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),  # L3
#     kl.Conv2D(kernel_size=(3, 3), filters=64, strides=(1, 1), padding='same', activation='relu'),  # L4
#     kl.BatchNormalization(),
#     kl.Conv2D(kernel_size=(3, 3), filters=64, strides=(1, 1), padding='same', activation='relu'),  # L5
#     kl.BatchNormalization(),
#     kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),  # L6
#     kl.Conv2D(kernel_size=(3, 3), filters=384, strides=(1, 1), padding='same', activation='relu'),  # L7
#     kl.BatchNormalization(),
#     kl.Conv2D(kernel_size=(3, 3), filters=384, strides=(1, 1), padding='same', activation='relu'),  # L8
#     kl.BatchNormalization(),
#     kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),  # L9
#     kl.Dense(2048, activation='softmax'),  # L10
#     kl.Dropout(0.5),
#     kl.Dense(2048, activation='softmax'),  # L11
#     kl.Dropout(0.5),
#     kl.Dense(2, activation='softmax'),  # L12
#     kl.Softmax()
# ])

model = tf.keras.models.Sequential([
    kl.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    kl.MaxPooling2D(2, 2),
    kl.Conv2D(64, (3, 3), activation='relu'),
    kl.MaxPooling2D(2, 2),
    kl.Conv2D(128, (3, 3), activation='relu'),
    kl.MaxPooling2D(2, 2),
    kl.Flatten(),
    kl.Dense(512, activation='relu'),
    kl.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=Adam(lr=0.01),
    metrics=['acc']
)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=val_generator,
    validation_steps=50,
    verbose=2
)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()

