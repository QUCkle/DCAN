# 这个文件是用来将csv文件还原成wav文件的，用来验证预处理过后的数据是否正确
import numpy as np
import soundfile as sf
import pandas as pd
from sklearn import preprocessing 
# 从CSV文件加载音频数据
filename = '/home/Q/dataset/DAIC-WOZ/output_8k/470.csv'
data = pd.read_csv(filename)
print(data.shape)
data = data.iloc[0:1,:]
# print(data.dtypes)
# 如果数据是一维的，将其重塑为二维数组
# if data.ndim == 1:
#     data = np.reshape(data, (1, -1))

data = data.astype(np.float32)

# print(data.dtypes)

print("Min value:", np.min(data))
print("Max value:", np.max(data))

Scaler=preprocessing.MinMaxScaler() 
data = Scaler.fit_transform(data.T).T
print(data.shape)
print(np.ravel(data).shape)
print("Min value:", np.min(data))
print("Max value:", np.max(data))
# 保存为WAV文件
sf.write('/home/Q/dataset/DAIC-WOZ/output1/470_return_2.wav',data.squeeze(), 8000)
