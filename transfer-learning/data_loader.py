from torchvision import datasets, transforms
import torch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 自定义的CSV文件格式的dataloader
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data_frame.iloc[idx, :-1].values.astype('float32')  # 除去最后一列标签的数据
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
   

# csv数据集
def load_data_csv(data_folder, batch_size, train, num_workers=0, **kwargs):
    
    transform = transforms.Compose([
        ToTensor()  # 转换为张量
    ])

    dataset = CSVDataset(csv_file=data_folder, transform=transform)
    # 创建源数据集的数据加载器
    # drop_last=True表示如果最后一个batch的样本数小于batch_size，则丢弃
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True if train else False, 
                            **kwargs, num_workers=num_workers,
                            drop_last=True if train else False)
    
    # kwargs前面的两个星号表示将kwargs中的所有键值对作为参数传递给get_data_loader函数
    data_loader = get_data_loader(dataset, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = 2
    return data_loader, n_class

   

# 提取好特征的数据集
def load_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    # 获取data_folder下的所有文件
    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    # 过滤出所有.csv文件
    csv_files = [f for f in files if f.endswith('.csv')]
    # 读取所有.csv文件
    data = [pd.read_csv(os.path.join(data_folder, f)) for f in csv_files]
    # print(data)
    
    # 将数据和标签分开
    features = [df.iloc[:, :-1].values for df in data]
    labels = [df.iloc[:, -1].values for df in data]
    
    # 计算均值和标准差
    mean = np.mean(np.concatenate(features), axis=0)
    std = np.std(np.concatenate(features), axis=0)
    
    # 使用MinMaxScaler进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_normalized = [scaler.fit_transform(f) for f in features]
    
    features=features_normalized 
    
    # 转换为Tensor
    features = [torch.tensor(f, dtype=torch.float32) for f in features]
    labels = [torch.tensor(l, dtype=torch.long) for l in labels]
    # 创建数据集
    datasets = [TensorDataset(f, l) for f, l in zip(features, labels)]
    # print(len(datasets))
    # 获取数据加载器
    data_loaders = [get_data_loader(d, batch_size=batch_size, shuffle=True if train else False, num_workers=num_workers, drop_last=not train, **kwargs) for d in datasets]
    # 获取类别数，方法是获取每个数据集的标签中不同的类别数
    n_classes = [len(torch.unique(l)) for l in labels]
    
    # 将数据和标签合并
    data_norm = [np.concatenate((f, l.reshape(-1, 1)), axis=1) for f, l in zip(features_normalized, labels)]
    data = np.array(data_norm,dtype=np.float32)
    data=np.squeeze(data)
    # print(len(data_loaders))
    # 返回数据加载器和类别数
    return data_loaders, n_classes,data


# 图片数据集
# **kwargs的作用是将kwargs中的所有键值对作为参数传递给load_data_img函数，kwargs是一个字典
def load_data_img(data_folder, batch_size, train, num_workers=0, **kwargs):
    
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    # kwargs前面的两个星号表示将kwargs中的所有键值对作为参数传递给get_data_loader函数
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class
    
    
    
def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0