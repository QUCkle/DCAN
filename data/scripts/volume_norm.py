import os
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
# 将所有音频文件都归一化到同一个音高
def normalize_volume(input_path, output_path, target_dBFS):
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in tqdm(filenames, desc="Processing files"):
            if filename.endswith('.wav'):
                input_file_path = os.path.join(dirpath, filename)
                relative_dirpath = os.path.relpath(dirpath, input_path)
                output_dirpath = os.path.join(output_path, relative_dirpath)
                os.makedirs(output_dirpath, exist_ok=True)
                output_file_path = os.path.join(output_dirpath, filename)

                sound = AudioSegment.from_file(input_file_path, format="wav")
                normalized_sound = match_target_amplitude(sound, target_dBFS)
                normalized_sound.export(output_file_path, format="wav")

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# 根据音高差异，给所有音频文件都加一点音高

def adjust_volume(gain, input_directory, output_directory, recursive=True):
    for dirpath, dirnames, filenames in os.walk(input_directory):
        if not recursive and dirpath != input_directory:
            continue
        for filename in tqdm(filenames, desc="Processing files"):
            if filename.endswith('.wav'):
                input_file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(dirpath, input_directory)
                output_dir_path = os.path.join(output_directory, relative_path)
                os.makedirs(output_dir_path, exist_ok=True)
                output_file_path = os.path.join(output_dir_path, filename)
                sound = AudioSegment.from_file(input_file_path, format="wav")
                adjusted_sound = sound.apply_gain(gain)
                adjusted_sound.export(output_file_path, format="wav")

def calculate_gain(mean_rms_source, mean_rms_target):
    return 20 * np.log10(mean_rms_target / mean_rms_source)


if __name__ == "__main__":
    # input_path = "/home/Q/dataset/DAIC-WOZ/clean_data"
    # output_path = "/home/Q/dataset/DAIC-WOZ/vol_norm_data"
    # target_dBFS = -20
    # normalize_volume(input_path, output_path, target_dBFS)
    
    mean_rms_zh = 0.0044509536
    mean_rms_en = 0.011742114
    gain = calculate_gain(mean_rms_zh, mean_rms_en)
    input_directory = "/home/Q/dataset/audio_lanzhou_2015/data_no_silence"  # 输入音频文件的目录
    output_directory = "/home/Q/dataset/audio_lanzhou_2015/data_vol_norm_2"  # 输出音频文件的目录
    recursive = True  # 是否递归处理子目录
    adjust_volume(gain, input_directory, output_directory, recursive)