import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from IPython.display import Audio
from scipy.signal import ricker, gaussian

import spleeter
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

def get_predictions(song_file, sep_name): 
    separator = Separator(sep_name)
    song, sr = librosa.load(song_file, sr=None)

    t_audio = np.arange(0, len(song)/sr, 1/sr)
    t_video = np.arange(0, len(song)/sr, 1/30)

    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(song_file, sample_rate=sr)
    prediction = separator.separate(waveform)

    # unpack as a list
    sources = [*prediction.keys()]

    # i want to create a new dictionary with keys source and values Audio(...)
    audio = {}
    for source in sources:
        audio[source] =  Audio(prediction[source].mean(axis=1), rate=sr)
    
    return prediction, sources, audio, sr

def analyze_power(prediction, sources, sr, filter, filter2):
    Sxx = {}
    v_fs = 30
    for source in sources:
        Sxx[source], _, t, img = plt.specgram(prediction[source].mean(axis=1), Fs=sr, NFFT=int(2*sr/v_fs), noverlap=int(sr/v_fs), mode='magnitude')

    power = {}
    for source in sources:
        power_temp = np.zeros(t.shape)
        for i,_ in enumerate(t):
                power_temp[i] = np.dot(Sxx[source][:,i], Sxx[source][:,i].T)

        # Each t bin is 0.03 seconds
        power_temp = np.abs(np.convolve(power_temp, filter[source], mode='same'))
        power_temp = np.convolve(power_temp, filter2[source], mode='same')

        power[source] = power_temp / power_temp.max()
    return power, t 

def create_animation(sources, power, colors):
    x = np.linspace(0,1,1000)
    fig = plt.figure(facecolor='k')
    ax = fig.add_subplot(111)

    p_comp = []
    for i, _ in enumerate(sources):
        pc, = ax.plot(x, np.sin(11*np.pi*x), colors[i])
        p_comp.append(pc)

    plt.xlim([0, 1])
    plt.ylim([-1, 1])
    plt.axis('off')

    pc_Hz =     [11, 7, 5, 3, 5]
    pc_phase =  [1,  1, 1, 1, 1]

    def animate(i):
        for ind, source in enumerate(sources):
            p_comp[ind].set_data(x, power[source][i]*np.sin(pc_Hz[ind]*np.pi*x - pc_phase[ind]*np.pi*i/100))

        return p_comp

    ani = FuncAnimation(fig, animate, range(len(power['vocals'])), interval=100/3)
    return ani