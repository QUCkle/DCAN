import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame.dropna(inplace=True)  # 删除包含NaN的行
        self.transform = transform
        self.scaler = StandardScaler()  # 创建StandardScaler实例
        # self.scaler = RobustScaler()  # 创建RobustScaler实例
        self.data_frame.iloc[:, :-1] = self.scaler.fit_transform(self.data_frame.iloc[:, :-1])  # 标准化数据
        # self.data_frame.iloc[:, :-1] = self.scaler.fit_transform(self.data_frame.iloc[:, :-1])  # 标准化数据


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        data = self.data_frame.iloc[idx, :-1].values.astype('float32')  # 除去最后一列标签的数据
        # data = (data - np.mean(data)) / np.std(data)  # Z-score normalization
        label = self.data_frame.iloc[idx, -1]  # 最后一列是标签
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class ToTensor(object):
    """将样本中的numpy数组转换为torch张量。"""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        return {'data': torch.from_numpy(data),
                'label': torch.tensor(label, dtype=torch.long)}