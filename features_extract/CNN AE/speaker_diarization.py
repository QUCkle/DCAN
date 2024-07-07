import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave
import re

# 这个函数的作用是将音频文件分割成多个音频文件，每个音频文件中只包含一个人的声音
def remove_silence(filename, out_dir, smoothing=1.0, weight=0.3, plot=False):
    partic_id = 'P' + filename.split('/')[-1].split('_')[0]  # PXXX
    partic_id = re.sub('\.wav$', '', partic_id)
    if is_segmentable(partic_id):
        # create participant directory for segmented wav files
        participant_dir = os.path.join(out_dir, partic_id).replace("\\","/")
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)

        #改变工作路径
        os.chdir(participant_dir)

        [Fs, x] = aIO.read_audio_file(filename)
        segments = aS.silence_removal(x, Fs, 0.020, 0.020,
                                     smooth_window=smoothing,
                                     weight=weight,
                                     plot=plot)

        for s in segments:
            seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, s[0], s[1])
            wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

        # concatenate segmented wave files within participant directory
        concatenate_segments(participant_dir, partic_id)


# toubled中的音频由于质量问题，不采用
def is_segmentable(partic_id):
    troubled = set(['P300', 'P305', 'P306', 'P308', 'P315', 'P316', 'P343',
                    'P354', 'P362', 'P375', 'P378', 'P381', 'P382', 'P385',
                    'P387', 'P388', 'P390', 'P392', 'P393', 'P395', 'P408',
                    'P413', 'P421', 'P438', 'P473', 'P476', 'P479', 'P490',
                    'P492'])
    return partic_id not in troubled

# 将参与者目录中的所有波形文件连接成一个单独的wav文件（移除了静音和其他说话者的部分），并写入到参与者的目录中，然后删除各个片段（当remove_segment=True时）。
def concatenate_segments(participant_dir, partic_id, remove_segment=True):
    
    infiles = os.listdir(participant_dir)  # list of wav files in directory
    outfile = '{}_no_silence.wav'.format(partic_id)

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        if remove_segment:
            os.remove(infile)

    output = wave.open(outfile, 'wb')
 
    output.setparams(data[0][0])


    for idx in range(len(data)):
        output.writeframes(data[idx][1])

    output.close()


if __name__ == '__main__':
    # 数据地址
    dir_name = '/home/Q/dataset/DAIC-WOZ/wav_data/'

    # 创建一个参与者文件夹的目录，其中包含他们的分割后的wav文件
    out_dir = '/home/Q/dataset/DAIC-WOZ/clean_data/'

    # 遍历dir_name中的wav文件，并创建分割后的wav文件
    for file in os.listdir(dir_name):
        if file.endswith('.wav'):
            filename = os.path.join(dir_name, file).replace("\\","/")
            remove_silence(filename, out_dir)
