# 0. 预设环境
import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import pandas as pd
import os
from scipy.stats import linregress
from tqdm import tqdm

# 1、均方根能量（Root mean square energy)
def calculate_rms(data, frame_size=1024):
    hop_size = int(frame_size * 0.5)
    rms = librosa.feature.rms(y=data, frame_length=frame_size, hop_length=hop_size).T[1:, 0]
    return np.mean(rms)

# rms_values = calculate_rms(data_0)

#2、过零率（Zero Crossing Rate）
def calculate_zero_crossing_rate(data):
    zcr = np.mean(librosa.feature.zero_crossing_rate(data, frame_length=1024, hop_length=512, center=True))
    zc = librosa.zero_crossings(data, pad=False, zero_pos=True)
    zc_sum = sum(zc)
    return zcr, zc_sum

# Call the function with your data_0
# zcr_values, zc_sum = calculate_zero_crossing_rate(data_0)

# 3、梅尔频率倒谱系数 librosa.feature.mfcc()

#4、voiceProb，从ACF（自相关函数）计算出的发声概率
def calculate_autocorrelation(data):
    acf = np.correlate(data, data, mode='full')
    acf /= np.max(np.abs(acf))
    voice_prob = 1 - acf
    return np.mean(voice_prob)

# Call the function with your data_0
# voice_probability_mean = calculate_autocorrelation(data_0)

# 5、F0+voicprobs(过零率)
def calculate_f0(data):
    f0, voiced_flag, voiced_probs = librosa.pyin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_no_nan = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0_no_nan)
    return f0_mean

# Call the function with your data_0
# pitch_mean = calculate_f0(data_0)

# 6、谱质心
# spec_cent = np.mean(librosa.feature.spectral_centroid(y=data_0, sr=fs))
# print(spec_cent)

# 7、频谱平坦度
def calculate_spectral_flatness(data):
    spec_flat = librosa.feature.spectral_flatness(y=data)
    spec_flat_mean = np.mean(spec_flat)
    return spec_flat_mean

# Call the function with your data_0
# spectral_flatness_mean = calculate_spectral_flatness(data_0)

# 8、频谱通量
def calculate_onset_strength(data, sr):
    onset_env = librosa.onset.onset_strength(y=data, sr=sr)
    onset_env_mean = np.mean(onset_env)
    return onset_env_mean

# Call the function with your data_0 and fs
# onset_strength_mean = calculate_onset_strength(data_0, fs)

# 9、轮廓线最值_幅值及其帧数
def calculate_window_max_min(audio_signal, window_length=2048, step_size=512):
    num_windows = (len(audio_signal) - window_length) // step_size + 1
    max_values = np.zeros(num_windows)
    min_values = np.zeros(num_windows)

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        window = audio_signal[start_idx:end_idx]
        # 这是每个窗口的最值
        max_values[i] = np.max(window)
        min_values[i] = np.min(window)

    max_position = np.argmax(max_values)
    min_position = np.argmin(min_values)
    return np.max(max_values), max_position, np.min(min_values), min_position

# Call the function with your data_0
# max_vals, max_pos, min_vals, min_pos = calculate_window_max_min(data_0)

# 10、轮廓线算术平均值_幅值
# mean=np.mean(data_0)

# 11、linregc1 轮廓线的线性近似的斜率以及偏移量以及二次误差
def calculate_linregc1(data_0):
    time = np.arange(len(data_0))
    audio_data = np.array(data_0)
    slope, intercept, r_value, p_value, std_err = linregress(time, audio_data)
    linear_approximation = slope * time + intercept
    linregerrQ = np.mean((audio_data - linear_approximation)**2)
    return slope, intercept, linregerrQ

# Call the function with your data_0
# slope, intercept, linregerrQ = calculate_linregc1(data_0)

# print("linregc1（线性近似的斜率）:", slope)
# print("linregerrQ（二次误差）:", linregerrQ)

# 12、13维MFCC特征
def ex_mfcc(data, label):
    global image_counter  # 声明使用全局变量
    
    # 计算MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=13)
    
    # 计算MFCCs的均值
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_mean = mfccs_mean.reshape(-1, 1)
    
    # 创建一个新的图像
    plt.figure(figsize=(10, 8))
    
    # 显示MFCCs
    librosa.display.specshow(mfccs, sr=fs)
    
    # 移除坐标轴
    plt.axis('off')
    
    # 保存图像
    path = f'/home/Q/dataset/audio_lanzhou_2015/pic/mfcc/{label}'
    os.makedirs(path, exist_ok=True)  # 确保目录存在
    plt.savefig(f'{path}/mfccs_{image_counter}.png', bbox_inches='tight', pad_inches=0)
    
    plt.close()
    # 更新图像编号
    image_counter += 1
    
    mfccs_mean=mfccs_mean.T
    
    return mfccs_mean

def de_stft(data,label):
    stft=librosa.stft(data, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')
    
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(stft,ref=np.max), y_axis='log', x_axis='time', ax=ax)
    plt.axis('off')
    path = f'/home/Q/dataset/audio_lanzhou_2015/pic/stft/{label}'
    
    os.makedirs(path, exist_ok=True)  # 确保目录存在
    plt.savefig(f'{path}/stft_{image_counter}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
# 将上面的特征提取函数集合
def distract(data_0,label):
    rms_values = calculate_rms(data_0)
    zcr_values, zc_sum = calculate_zero_crossing_rate(data_0)
    voice_probability_mean = calculate_autocorrelation(data_0)
    pitch_mean = calculate_f0(data_0)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=data_0, sr=fs))
    spectral_flatness_mean = calculate_spectral_flatness(data_0)
    onset_strength_mean = calculate_onset_strength(data_0, fs)
    max_vals, max_pos, min_vals, min_pos = calculate_window_max_min(data_0)
    mean=np.mean(data_0)
    slope, intercept, linregerrQ = calculate_linregc1(data_0)
    
    # mfccs_mean = ex_mfcc(data_0,label)
    # de_stft(data_0,label)
    # 将 mfccs_mean 转换为一维数组
    # mfccs_mean = mfccs_mean.flatten()

    # 将 pitch_mean 添加到 mfccs_mean 的右边
    # result = np.hstack((mfccs_mean, pitch_mean))

    return [rms_values, zcr_values, zc_sum, voice_probability_mean, pitch_mean, spec_cent, spectral_flatness_mean, onset_strength_mean, max_vals, max_pos, min_vals, min_pos, mean, slope, intercept, linregerrQ]
    # return result
    

class fe_diatract():
    def __init__(self, sample_rate, time, data_dir,save_dir):
        self.sample_rate = sample_rate
        # 取样时间
        self.time = time
        self.data_dir_origin = data_dir
        self.save_dir_origin = save_dir

    def ex_features(self,label):
        

        self.data_dir=self.data_dir_origin + '/' + label
        self.save_dir=self.save_dir_origin + '/' + label
        
        self.csv_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        feature_all = []
        

        for i in tqdm(self.csv_files, desc="Processing 1 files"):
            data_1 = pd.read_csv(i)
            data_1 = data_1.iloc[:, 1:]
            for j in range(data_1.shape[0]):
                print(j)
                data_1_p = data_1.iloc[j, :]
                data_1_p = np.ravel(data_1_p)
                feature_1 = distract(data_1_p,label=label)
                feature_all.append(feature_1)
                           
            file_path_1 = self.save_dir + '/' + os.path.basename(i)
            os.makedirs(self.save_dir,exist_ok=True)
            pd.DataFrame(feature_all).to_csv(file_path_1, index=False)

        return feature_all


if __name__ == '__main__':
    fs=16000  
    time=4
    # 全局变量，用于跟踪当前的图像编号
    image_counter = 0
    dis_fe = fe_diatract(16000, time=4, data_dir='/home/Q/dataset/audio_lanzhou_2015',save_dir='/home/Q/Diploma_thesis/Diastract-fe/data/result/ZH')
    dis_fe.ex_features(label='HC')
    dis_fe.ex_features(label='MDD')
