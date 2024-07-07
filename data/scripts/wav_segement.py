import os
import pandas as pd
import librosa
import numpy as np

# 切割音频数据,并且进行标签处理,重采样

# 读取Excel文件
excel_file = "/home/Q/dataset/audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx"
df = pd.read_excel(excel_file)

# 设定参数
sample_rate = 16000
window_size = 4 * sample_rate
hop_length = int(1 * sample_rate)
max_segments = 400  # 最大时间窗数量
# 原始数据路径
data_path = '/home/Q/dataset/audio_lanzhou_2015/data_vol_norm_2'
# 处理每个子文件夹下的音频文件
for subdir, dirs, files in os.walk(data_path):
    if files:
        subject_id = os.path.basename(subdir)
        if subject_id.isdigit():  # 检查是否是子文件夹
            label_row = df[df['subject id'] == int(subject_id)]
            if not label_row.empty:
                label = label_row['type'].iloc[0]  # 获取标签
                audio_data = np.array([])

                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(subdir, file)
                        y, sr = librosa.load(file_path, sr=sample_rate)
                        audio_data = np.concatenate((audio_data, y))

                # 保证音频数据长度不超过500个时间窗
                max_audio_length = min(len(audio_data), max_segments * hop_length)

                # 切割音频数据
                num_segments = (max_audio_length - window_size) // hop_length + 1

                save_path = f"/home/Q/dataset/audio_lanzhou_2015/data_resample/norm_2/16k/{label}/{subject_id}.csv"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'a') as f:
                    for i in range(num_segments):
                        start = i * hop_length
                        segment = audio_data[start: start + window_size]
                        if len(segment) == window_size:
                            np.savetxt(f, [segment], delimiter=",")
                        if i >= max_segments - 1:  # 达到500个时间窗后停止处理
                            break
