import os
import librosa

# 定义函数来加载 WAV 文件并检查采样率
def load_wav(file_path):
    try:
        # 使用 librosa 加载 WAV 文件
        y, sr = librosa.load(file_path, sr=16000)
        # 如果采样率不是 16k，则删除文件并输出其路径
        if sr != 16000:
            os.remove(file_path)
            print("Deleted:", file_path)
    except Exception as e:
        # 加载过程中出错，删除文件并输出其路径
        os.remove(file_path)
        print("Error:", e, "-", file_path)

# 定义函数来遍历文件夹及其子文件夹
def traverse_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为 WAV 格式
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                load_wav(file_path)

# 指定要遍历的文件夹路径
folder_path = "/home/Q/dataset/audio_lanzhou_2015/data_no_silence"

# 开始遍历文件夹
traverse_folder(folder_path)
