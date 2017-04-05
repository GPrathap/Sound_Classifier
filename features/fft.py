import librosa
import numpy as np

from utils import feature_extractor as utils

class FFT:
    def __init__(self, audio, dependencies=None, number_of_bins=1024, frame=2048, sampling_rate=44000, is_raw_data=True):
        self.audio = audio
        self.dependencies = dependencies
        self.frame = frame
        self.sampling_rate = sampling_rate
        self.number_of_bins = number_of_bins
        self.is_raw_data =  is_raw_data
        if self.is_raw_data:
            self.frames = int(np.ceil(len(self.audio.data) / 1000.0 * self.sampling_rate / self.frame))
        else:
            self.frames = 1


    def __enter__(self):
        print "Initializing fft calculation..."

    def __exit__(self, exc_type, exc_val, exc_tb):
        print "Done with calculations..."

    def compute_fft(self):
        self.fft = []
        self.logamplitude = []
        for i in range(0, self.frames):
            if self.is_raw_data:
                current_frame = utils._get_frame(self.audio, i, self.frame)
            else:
                current_frame = self.audio.data
            ps = np.abs(np.fft.fft(current_frame, self.number_of_bins))
            self.fft.append(ps)
            self.logamplitude.append(librosa.logamplitude(ps ** 2))
        if self.is_raw_data:
            self.fft = np.asarray(self.fft)
            self.logamplitude = np.asarray(self.logamplitude)
        else:
            self.fft = np.asarray(self.fft)[0]
            self.logamplitude = np.asarray(self.logamplitude)[0]

    def get_fft_spectrogram(self):
        return self.fft


    def get_logamplitude(self):
        return self.logamplitude
