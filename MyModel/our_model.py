import tensorflow as tf


def get_model():
    #输入层
    input_image = tf.keras.layers.Input(shape=(224, 224, 3), dtype="float32")
    #卷积层(输出通道数，卷积核大小，卷积步长)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2,use_bias=False)(input_image)
    #归一化层
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    #激活函数
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAvgPool2D(name="avgPool")(x)  # 1*1*128 特征降维
    # DropOut层(失活比例)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    #全连接层(全连接层个数,激活函数)
    x = tf.keras.layers.Dense(units=100,activation="relu")(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(2, name="logits")(x)
    #softmax层，保证输出的结果是[0,1]的概率值
    predict = tf.keras.layers.Softmax()(x)
    #构建模型
    model = tf.keras.Model(inputs=input_image, outputs=predict,name='tb_model')
    return model


