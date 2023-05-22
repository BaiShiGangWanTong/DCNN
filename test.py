import glob
from time import time
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import tensorflow as tf
from MyModel.SBNN import get_scnn
from tool import genConfusionMatrix
from tool import roc_auc


#加载测试集
rootpath=r"G:\dataset\TB_Database22\test"
assert os.path.exists(rootpath), "cannot find {}".format(rootpath)
tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)


im_height = 224
im_width = 224
batch_size = 2

#创建模型
model =get_model()
weights_path = r'./weights/epoch0-acc0.500-loss0.717_newModel.ckpt'
assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)

#加载权重
model.load_weights(weights_path)

#测试集数据归一化
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#创建数据集生成器，打包测试集数据
test_data_gen = test_image_generator.flow_from_directory(directory=rootpath,
                                                         target_size=(im_height,im_width),
                                                         batch_size=batch_size,
                                                         class_mode='sparse',
                                                         shuffle=False)
#获取所有的测试集样本数
total_test = test_data_gen.n

#测试过程验证集进度条
val_bar = tqdm(range(total_test // batch_size))

#预测类别
result = np.array([],dtype=int)
#真实类别
label = np.array([],dtype=int)

times = 0.0

for step in val_bar:
    #加载测试数据
    test_images, test_labels = next(test_data_gen)
    start = time()
    #将一个batch数据送入网络，获得输出
    data_numpy = model(test_images, training=False)
    #记录测试的时间
    times+=(time()-start)
    #转化成numpy格式
    data_numpy = data_numpy.numpy()

    #获取预测结果
    result = np.append(result,data_numpy.argmax(axis=1))
    #获得真实标签
    label = np.append(label,test_labels)

end = time()

print("耗费时间:",times/total_test)
#计算所需要的指标
label = label.astype(np.int8)
matrix = genConfusionMatrix(2,result,label)
matrix_se = matrix[1][1]/(matrix[1][0]+matrix[1][1])
matrix_sp = matrix[0][0]/(matrix[0][1]+matrix[0][0])
matrix_acc = (matrix[0][0]+matrix[1][1])/np.sum(matrix)
matrix_auc = roc_auc(label,result)
matrix_pre = matrix[1][1]/(matrix[0][1]+matrix[1][1])
matrix_youden = matrix_se+matrix_sp-1
print("混淆矩阵：")
print(matrix)
print("matrix_se",matrix_se)
print("matrix_sp",matrix_sp)
print("matrix_auc",matrix_auc)
print("matrix_acc",matrix_acc)
print("matric_pre",matrix_pre)
print("约登指数",matrix_youden)
print("weights_path:", weights_path)

