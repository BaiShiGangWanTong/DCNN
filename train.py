import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"#使用cpu进行训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import tensorflow as tf
from tqdm import tqdm
from MyModel import SBNN

def main():
    #设置数据集路径
    image_path = "B:/PycharmProjects/DCNN/dataset"
    #训练集路径
    train_dir = os.path.join(image_path, "train")
    #验证集路径
    validation_dir = os.path.join(image_path, "val")
    #权重路径
    weight_path = "weights"
    assert os.path.exists(weight_path), "cannot find {}".format(weight_path)
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

#参数配置
    #送入网络的图像大小
    im_height = 224
    im_width = 224
    #一次送入多少张图像到网络中
    batch_size = 4
    #训练总次数
    epochs = 100
    #学习率
    lr = 0.01
    #训练数据增强器
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                               rescale=1./255,
                                               vertical_flip=True,
                                               rotation_range=6,
                                               brightness_range=[0.1, 2])
    #验证数据增强器
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    #生成训练集数据加载器
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    #生成验证集数据加载器
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=True,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    #训练样本总个数
    total_train = train_data_gen.n
    #测试集样本总个数
    total_val = val_data_gen.n

    # 训练数据类别标签
    class_indices = train_data_gen.class_indices

    # 自定义类别标签名称
    inverse_dict = dict((val, key) for key, val in class_indices.items())

    #标签写入文件
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))
    #创建模型
    model = SBNN.get_sbnn()

    #打印模型结构
    model.summary()

    # 损失函数:交叉熵
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    #优化器
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    #整体损失采用均值损失
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    #训练集精度计算
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    #验证集损失
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    #验证集精度计算
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            #获得模型输出
            output = model(images, training=True)
            #计算损失
            loss = loss_object(labels, output)
        #计算梯度（求导）
        gradients = tape.gradient(loss, model.trainable_variables)
        #应用梯度（反向传播）
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 对一个batch_size的loss求均值
        train_loss(loss)
        # 对一个batch_size的预测求准确率
        train_accuracy(labels, output)


    @tf.function
    def val_step(images, labels):
        output = model(images, training=False)
        loss = loss_object(labels, output)
        val_loss(loss)
        val_accuracy(labels, output)

    best_val_acc = 0.
    trainloss=[]
    trainaccuracy = []
    valloss=[]
    valaccuracy = []

    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # 训练一个epoch的迭代次数
        count = range(total_train // batch_size)
        train_bar = tqdm(count)
        for step in train_bar:
            # 每个step从训练集中取出一个打包好的数据
            images, labels = next(train_data_gen)
            train_step(images, labels)
            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # 训练完成，进行验证
        val_bar = tqdm(range(total_val // batch_size))
        for step in val_bar:
            test_images, test_labels = next(val_data_gen)
            val_step(test_images, test_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())


        #一个epoch之后，记录loss和acc
        trainloss.append(train_loss.result().numpy())
        trainaccuracy.append(train_accuracy.result().numpy())
        valloss.append(val_loss.result().numpy())
        valaccuracy.append(val_accuracy.result().numpy())

        # 仅仅保存最优的权重
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights(weight_path+"\epoch{}-acc{:.3f}-loss{:.3f}_newModel.ckpt".format(
                epoch,val_accuracy.result(),val_loss.result()
            ),save_format='tf')

    print("trainloss:{}".format(trainloss))
    print("trainaccuracy:{}".format(trainaccuracy))
    print("valloss:{}".format(valloss))
    print("valaccuracy:{}".format(valaccuracy))


if __name__ == '__main__':
    main()

