# cnn autoncoder model
# 这个文件的作用是利用训练好的模型，对数据进行编码，提取特征
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, AveragePooling1D,Reshape,UpSampling1D
from keras.models import Model, Sequential
from keras import initializers
from keras.layers import BatchNormalization

#CNN autoencoder
def CNN_encoder(input,dims):
  ## encoding part
  initializer = tf.keras.initializers.HeNormal()
  # 四层卷积层，再加池化层压缩数据最后再flat展开，一维卷积
  # 参数分别为卷积核个数即输出维度，卷积核大小，激活函数,padding是当卷积核超出边界时的处理方式,same是补0,padding的其他参数还有valid，不补0
  layer = Conv1D(64, kernel_size=3, padding="same", activation="relu")(input)
  layer = BatchNormalization()(layer)
  layer = MaxPooling1D(pool_size=2)(layer)
  layer = Dropout(0.25)(layer)
  layer = Conv1D(128, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = MaxPooling1D(pool_size=2)(layer)
  layer = Dropout(0.25)(layer)
  layer = Conv1D(256, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = MaxPooling1D(pool_size=2)(layer)
  layer = Dropout(0.25)(layer)
  layer = Conv1D(512, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = MaxPooling1D(pool_size=2)(layer)
  layer = Dropout(0.25)(layer)

  layer = Flatten()(layer)
  # 这一层的作用是将输入展平。不影响批量大小。encoded的shape是(?, 512)
  encoded = Dense(dims, activation="relu")(layer)
  return encoded

# 解码函数，通过卷积层+上采样层，还原数据
def CNN_decoder(encoded):
  ## decoding part
  initializer = tf.keras.initializers.HeNormal()
  layer = Dense(250* 512, activation="relu", use_bias=False)(encoded)
  layer = Reshape((250, 512))(layer)
  layer = Conv1D(512, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = UpSampling1D(size=2)(layer)
  layer = Conv1D(256, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = UpSampling1D(size=2)(layer)
  layer = Conv1D(128, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = UpSampling1D()(layer)
  layer = Conv1D(64, kernel_size=3, padding="same", activation="relu")(layer)
  layer = BatchNormalization()(layer)
  layer = UpSampling1D()(layer)
  decoded = Conv1D(1, kernel_size=3, padding="same", activation="sigmoid")(layer)

  return decoded
def CAE_model(Train_X,dims):
  CNNinput_layer = Input(shape=(Train_X.shape[1],1))
  # 编码后的数据
  cnn_x = CNN_encoder(CNNinput_layer,dims)
  # 解码后的数据
  cnn_x_hat = CNN_decoder(cnn_x)
  # 把编码和解码拼接成一个模型
  CNNmodel = Model(CNNinput_layer, cnn_x_hat)
  CNN_enconder = Model(CNNinput_layer, cnn_x, name="encoder")
  return CNNmodel,CNN_enconder

def CAE_hiddenrepresentation(data,method_enconder):
        """Transform the vector.
        From original dimensions to latent dimensions.
        Parameters
        ----------
        data : array-like  (n_samples, n_features)
            The input data to use in transform process.
        Returns
        -------
        _ : array-like  (n_samples, value_encoding_dim)
            The data transformed to latent dimensions format.
        """
        return method_enconder.predict(data)
      
# 截取编码层
def CNNextractrep(model):
  hidden_representation = Sequential()
  for i in range(19):  # Adjust this number based on the total number of layers in your model
    hidden_representation.add(model.layers[i])
  return hidden_representation


import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing 
from sklearn import svm
from tensorflow import keras
from sklearn.metrics import plot_confusion_matrix
from keras.callbacks import History
from sklearn.metrics import confusion_matrix
import re
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import ClusterCentroids


sns.set(style="whitegrid")
np.random.seed(203)
# drive.mount('/content/drive')

encodding_dim=700
# 1k采样，4s


#读取数据
data=pd.read_csv('/home/Q/dataset/DAIC-WOZ/output_1k/470.csv')
#数据预处理
#---------before training perform Max-Min scalling
pre_data=data
data = data.iloc[0:1,:]

CAE_model, _ = CAE_model(data, encodding_dim)

# 加载已训练模型的权重
CAE_model.load_weights("/home/Q/Diploma_thesis/CAE_ADD-main/models/modeltotalCAE_norm_1k.h5")


#加载编码器
LatentRepresenation=CNNextractrep(CAE_model)

# 先用训练集，fit_transform,修改Scaler内部参数，再用测试集transform
new_Train_total=pd.read_csv('/home/Q/Diploma_thesis/CAE_ADD-main/data/new_Train_total_CAE_1k.csv')

#---------before training perform Max-Min scalling
Scaler=MinMaxScaler()  # 实例化一个MinMaxScaler对象，用于最大最小值缩放
x_train = new_Train_total.drop(["Class"], axis=1)  # 从new_Train_total中删除"Class"列，得到训练数据
y_train = new_Train_total["Class"].values  # 获取训练数据的标签

x_scaled_train = Scaler.fit_transform(x_train.values)  # 使用MinMaxScaler对训练数据进行最大最小值缩放


# 编码函数,也就是将数据转换为潜在空间,即特征提取
def encode(data,Scaler):
  # audio全变成0了，我得看看是不是数据的问题
  # .values,将DataFrame对象转换为Numpy的ndarray对象。这是因为MinMaxScaler的fit_transform方法需要一个ndarray对象
  data = Scaler.transform(data.values)
  data = np.expand_dims(data, axis=2)
  encoded_features=LatentRepresenation.predict(data)
  return encoded_features
  
# 数据地址 
data_dir='/home/Q/dataset/DAIC/raw_data/1k'

# 特征输出地址
output_dir = '/home/Q/dataset/DAIC/features/CAE/1k'



# data_path = '/home/Q/Diploma_thesis/CAE_ADD-main/data/new_Test_total_CAE_1k.csv'
# data = pd.read_csv(data_path, header=None)
# label = data.iloc[:, -1]
# data_val = data.iloc[:, :-1]
# encode_data = encode(data_val)
# encode_data = pd.DataFrame(encode_data)
# encode_data['label'] = label
# encode_data.to_csv('/home/Q/dataset/transfer-learning/trans-language/CAE/en_test.csv', index=False, header=False)

# 读取data_dir目录中的每个csv文件
for label in ['0', '1']:
  label_dir = os.path.join(data_dir, label)
  feature_dir = os.path.join(output_dir, label)
  os.makedirs(feature_dir, exist_ok=True)
  
  for file_name in os.listdir(label_dir):
    if file_name.endswith('.csv'):
      file_path = os.path.join(label_dir, file_name)
      df = pd.read_csv(file_path,header=None)
      
      # 创建一个空的DataFrame来存储编码后的特征
      encoded_features = pd.DataFrame()

      # # .values,将DataFrame对象转换为Numpy的ndarray对象。这是因为MinMaxScaler的fit_transform方法需要一个ndarray对象
      # df = Scaler.fit_transform(df.values)
      
      encoded_features = pd.DataFrame(encode(df,Scaler))

      # 将编码后的特征连接并保存到新的csv文件中
      output_file_path = os.path.join(feature_dir, file_name)
      encoded_features.to_csv(output_file_path, index=False)

