import pandas as pd
from imblearn.over_sampling import SMOTE

def balance_dataset(input_file, output_file, augmentation_method):
    # 加载数据集
    data = pd.read_csv(input_file, header=None)
    
    # 打印增强前的标签数量
    print("Before augmentation and dropna, label counts:")
    print(data.iloc[:, -1].value_counts())
    

    # # 删除包含 NaN 的行
    data.dropna(inplace=True)

    # 打印增强前的标签数量
    print("Before augmentation, label counts:")
    print(data.iloc[:, -1].value_counts())

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 使用传入的数据增强方法进行数据增强
    X_balanced, y_balanced = augmentation_method.fit_resample(X, y)

    # 合并增强后的特征和标签
    balanced_data = pd.concat([pd.DataFrame(X_balanced), pd.DataFrame(y_balanced)], axis=1)

    # 打印增强后的标签数量
    print("After augmentation, label counts:")
    print(balanced_data.iloc[:, -1].value_counts())

    # 保存增强后的文件
    balanced_data.to_csv(output_file, index=False, header=False)
    
    
balance_dataset(
    input_file="/home/Q/dataset/transfer-learning/trans-language/手动/EN_test.csv",
    output_file="/home/Q/dataset/transfer-learning/trans-language/手动/EN_test_balanced.csv",
    augmentation_method=SMOTE(random_state=42)
)