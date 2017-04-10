import os

import seaborn as sb

from features.fft import FFT
from features.generic_type import EMG
from features.mean import Mean
from features.mfcc import MFCC
from features.zcr import ZCR
from manager import FeatureManager
from utils.Audio import Audio
import numpy as np

sb.set(style="white", palette="muted")

import random
random.seed(20150420)

class Clip:

    def __init__(self, buffer=None, filename=None, file_type=None, is_raw_data=True, number_of_bins=0, frame=128, sampling_rate=250):
        self.is_raw_data = is_raw_data
        self.frame = frame
        self.sampling_rate = sampling_rate
        if self.is_raw_data:
            self.filename = os.path.basename(filename)
            self.path = os.path.abspath(filename)
            self.directory = os.path.dirname(self.path)
            self.category = self.directory.split('/')[-1]
            self.audio = Audio(self.path, file_type)
            self.number_of_bins = number_of_bins

        else:
            self.audio = Audio(is_raw_data=self.is_raw_data, data=buffer)
            self.number_of_bins = int(np.ceil(len(self.audio.data)/2))

        with self.audio as audio:
            self.featureManager = FeatureManager()
            self.featureManager.addRegisteredFeatures(FFT(self.audio,None,self.number_of_bins, self.frame,
                                                          self.sampling_rate, is_raw_data=self.is_raw_data), "fft")

            self.featureManager.addRegisteredFeatures(EMG(self.audio,None,number_of_bins =number_of_bins,
                                                          frame=self.frame, sampling_rate=self.sampling_rate,
                                                          is_raw_data=self.is_raw_data), "emg")

            self.featureManager.getRegisteredFeature("fft").compute_fft()
            self.featureManager.getRegisteredFeature("emg").compute_hurst()
            # self.featureManager.getRegisteredFeature("emg").compute_embed_seq(Tau, D)
            # self.featureManager.getRegisteredFeature("emg").compute_bin_power(Band)
            # self.featureManager.getRegisteredFeature("emg").compute_pfd(D)
            # self.featureManager.getRegisteredFeature("emg").compute_hfd(Kmax)
            # self.featureManager.getRegisteredFeature("emg").compute_hjorth(D)
            # self.featureManager.getRegisteredFeature("emg").compute_spectral_entropy()
            # self.featureManager.getRegisteredFeature("emg").compute_svd_entropy(W)
            # self.featureManager.getRegisteredFeature("emg").compute_ap_entropy(M, R)
            # self.featureManager.getRegisteredFeature("emg").compute_samp_entropy(M, R)
            # self.featureManager.getRegisteredFeature("emg").compute_dfa(Ave, L)
            # self.featureManager.getRegisteredFeature("emg").compute_permutation_entropy()
            # self.featureManager.getRegisteredFeature("emg").compute_LLE(t)

            self.feature_list = self.featureManager.getRegisteredFeatures()

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)

    def get_feature_vector(self):
        self.featureManager.getRegisteredFeature("emg").get_hurst()
        return self.featureManager.getRegisteredFeature("fft").get_logamplitude()




































