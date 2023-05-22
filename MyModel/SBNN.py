import tensorflow as tf


def get_sbnn():
    input_image = tf.keras.layers.Input(shape=(224, 224, 3), dtype="float32")
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=32, strides=1, padding='same', activation='relu')(input_image)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=32, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=64, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=384, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, filters=384, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = tf.keras.layers.Dense(2048, activation='softmax')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(2048, activation='softmax')(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(2, name="logits", activation='softmax')(x)
    predict = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=input_image, outputs=predict, name='sbnn_model')
    return model

