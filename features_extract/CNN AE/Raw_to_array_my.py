import numpy as np
import os
import librosa
from librosa import display
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave
import re
import ffmpeg
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm  # 导入tqdm库

def make_frames(filename, folder, frame_length, step):
    num_frames = 150  
    filename = '/home/Q/dataset/DAIC-WOZ/clean_data/' + folder + '/' + filename  
    data, sample_rate = librosa.load(filename, sr=1000, mono=True)      
    frame_length = frame_length * sample_rate  
    step = int(step * sample_rate)  
    colnames = range(0, frame_length)  
    total_df = []  
    
    for i in range(0, num_frames):
        start_idx = i * step
        end_idx = start_idx + frame_length
        if end_idx <= len(data):
            temp = np.array([data[start_idx:end_idx]])
            df = pd.DataFrame(temp, columns=colnames)
            total_df.append(df)
    
    new_df = pd.concat(total_df)  
    return new_df  

def make_frames_folder(folders,frame_length,step):
    for folder in tqdm(folders):  # 使用tqdm显示进度条
        files = os.listdir(dir_name + '/' + folder)
        for file in tqdm(files):  # 使用tqdm显示进度条
            res = make_frames(file,folder,frame_length,step)
            partic_id = os.path.basename(folder)
            nameid = re.sub('P', '', partic_id)
            nameid = int(nameid)
            filenm = str(nameid) + '.csv'
            output = out_dir + filenm
            res.to_csv(output, index=False)
            del res

if __name__ == '__main__':
    # 时间窗大小
    frame_length = 4
    # 步长
    step=3
    dir_name = '/home/Q/dataset/DAIC-WOZ/clean_data'
    folder = dir_name
    out_dir = '/home/Q/dataset/DAIC-WOZ/output_1k/'
    files = os.listdir(dir_name)
    folders = os.listdir(dir_name)

    make_frames_folder(folders, frame_length,step)