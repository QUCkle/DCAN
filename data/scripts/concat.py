
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def concat_csv_files(path, drop_first_row=True, num_rows=None, output_path='', add_label=0, recursive=True, split_dataset=False):
    """
    将多个CSV文件连接成一个DataFrame，并将其保存为新的CSV文件。

    参数:
    - path (str): 包含CSV文件的目录路径。
    - drop_first_row (bool, optional): 是否删除每个CSV文件的第一行。默认为True。
    - num_rows (int, optional): 每个CSV文件包含的行数。默认为None（所有行）。
    - output_path (str, optional): 将连接后的DataFrame保存为新的CSV文件的路径。默认为空字符串。
    - add_label (int, optional): 标签类型选择，0表示不添加标签，1表示以文件名csv为标签，2表示以该文件的上层文件夹名字为标签。默认为0。
    - recursive (bool, optional): 是否递归搜索子目录中的CSV文件。默认为True。
    - split_dataset (bool, optional): 是否将连接后的DataFrame划分为训练集和测试集。默认为False。
    - split_by_file (bool, optional): 是否将整个CSV文件作为训练集或测试集。默认为False。
    """
    # 初始化两个空的数据框来存储训练集和测试集的数据
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # 根据recursive的值决定是否递归遍历文件夹下的所有csv文件
    if recursive:
        # 递归遍历文件夹下的所有csv文件
        csv_files = [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith('.csv')]
    else:
        # 只遍历当前文件夹下的csv文件
        csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

    if split_dataset:
        # 划分训练集和测试集的文件
        train_files, test_files = train_test_split(csv_files, test_size=0.3)

        # 遍历训练集的文件，将数据添加到train_data中
        for file in tqdm(train_files):
            data = pd.read_csv(file, header=None)
            if drop_first_row:
                data = data.drop(0)
            if num_rows is not None:
                data = data.head(num_rows)
            if add_label == 1:
                label = int(os.path.splitext(os.path.basename(file))[0])  # 获取文件名（去除.csv后的部分）
            elif add_label == 2:
                label = int(os.path.basename(os.path.dirname(file)))  # 获取上层文件夹名字
            else:
                label = None
            if label is not None:
                data['label'] = label  # 添加标签列
            train_data = pd.concat([train_data, data], axis=0)

        # 遍历测试集的文件，将数据添加到test_data中
        for file in tqdm(test_files):
            data = pd.read_csv(file, header=None)
            if drop_first_row:
                data = data.drop(0)
            if num_rows is not None:
                data = data.head(num_rows)
            if add_label == 1:
                label = int(os.path.splitext(os.path.basename(file))[0])  # 获取文件名（去除.csv后的部分）
            elif add_label == 2:
                label = int(os.path.basename(os.path.dirname(file)))  # 获取上层文件夹名字
            else:
                label = None
            if label is not None:
                data['label'] = label  # 添加标签列
            test_data = pd.concat([test_data, data], axis=0)


        # 将训练集和测试集保存为新的csv文件
        train_data.to_csv(output_path.replace('.csv', '') + '_train.csv', index=False, header=False)
        print(output_path.replace('.csv', '') + '_train.csv')
        test_data.to_csv(output_path.replace('.csv', '') + '_test.csv', index=False, header=False)
    else:
        # 原有的处理逻辑
        for file in tqdm(csv_files):
            data = pd.read_csv(file,header=None)
            if drop_first_row:
                data = data.drop(0)
            if num_rows is not None:
                data = data.head(num_rows)
            if add_label == 1:
                label = int(os.path.splitext(os.path.basename(file))[0])  # 获取文件名（去除.csv后的部分）
            elif add_label == 2:
                label = int(os.path.basename(os.path.dirname(file)))  # 获取上层文件夹名字
            else:
                label = None
            if label is not None:
                data['label'] = label  # 添加标签列
            train_data = pd.concat([train_data, data], axis=0)

        # 将结果保存为一个新的csv文件
        train_data.to_csv(output_path, index=False, header=False)

if __name__ == '__main__':
    # 指定要合并的文件夹路径
    # path = '/home/Q/dataset/audio_lanzhou_2015/data_resample/1k'
    # output_path='/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k/zh_all.csv'
    # # 调用concat_csv_files函数，合并指定路径下的所有csv文件，并将结果保存为一个新的csv文件
    # concat_csv_files(path, num_rows = 200 ,output_path = output_path, add_label=2,recursive=True,split_dataset=True)
    
    
    path = '/home/Q/dataset/audio_lanzhou_2015/features/手动/csv'
    output_path='/home/Q/dataset/audio_lanzhou_2015/features/手动/zh_all.csv'
    # 调用concat_csv_files函数，合并指定路径下的所有csv文件，并将结果保存为一个新的csv文件
    concat_csv_files(path, num_rows = 150 ,drop_first_row=True,output_path = output_path, add_label=2,recursive=True,split_dataset=True)
    
    # # 指定另一个要合并的文件夹路径
    # path = '/home/Q/dataset/audio_lanzhou_2015/features/CAE/1k/MDD'
    # # 再次调用concat_csv_files函数，合并指定路径下的所有csv文件，并将结果保存为一个新的csv文件
    # concat_csv_files(path, output_path='/home/Q/dataset/audio_lanzhou_2015/features/CAE/1k/MDD_all.csv', add_label=True)