import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def remove_silence(src_dir, dst_dir):
    """
    递归地移除指定目录及其子目录下所有wav文件的静音部分，并将结果保存到另一个目录。

    参数:
    src_dir (str): 源目录路径。
    dst_dir (str): 目标目录路径。
    """

    # 获取所有.wav文件的列表
    wav_files = [os.path.join(root, filename) 
                 for root, dirs, files in os.walk(src_dir) 
                 for filename in files 
                 if filename.endswith(".wav")]

    # 使用tqdm来显示进度条
    for wav_file in tqdm(wav_files, desc="Processing .wav files"):
        # 加载音频文件
        audio = AudioSegment.from_wav(wav_file)

        # 使用split_on_silence函数来分割音频文件
        chunks = split_on_silence(audio, min_silence_len=600, silence_thresh=-60, keep_silence=400)

        # 将所有非静音部分连接起来
        nonsilent_audio = sum(chunks, AudioSegment.empty())

        # 创建相应的目标目录
        relative_path = os.path.relpath(os.path.dirname(wav_file), src_dir)
        dst_subdir = os.path.join(dst_dir, relative_path)
        os.makedirs(dst_subdir, exist_ok=True)

        # 保存到目标目录
        nonsilent_audio.export(os.path.join(dst_subdir, os.path.basename(wav_file)), format="wav")

if __name__ == "__main__":
    src_dir = "/home/Q/dataset/audio_lanzhou_2015/data_origin"
    dst_dir = "/home/Q/dataset/audio_lanzhou_2015/data_no_silence"

    remove_silence(src_dir, dst_dir)