# import json
# import os
#
# import seaborn as sb
#
# from features.fft import FFT
# from features.generic_type import EMG
# from features.mean import Mean
# from features.mfcc import MFCC
# from features.zcr import ZCR
# from manager import FeatureManager
# from utils.Audio import Audio
#
# sb.set(style="white", palette="muted")
#
# import random
# random.seed(20150420)
# from scipy.signal import butter, filtfilt
# import numpy as np
# import matplotlib.pyplot as plt
#
# import pandas as pd
# from pandas import DataFrame, Series
# from scipy.signal import butter, filtfilt
# from pandas import DataFrame, Series
# from scipy import signal
#
# class Clip:
#
#     def __init__(self, config, buffer=None, filename=None, file_type=None):
#         self.is_raw_data = eval(config["is_raw_data"])
#         self.frame_size = int(config["window_size"])
#         self.sampling_rate = int(config["sampling_rate"])
#         # self.project_path = str(config["project_file_path"])
#         self.project_path = "/home/runge/openbci/git/OpenBCI_Python"
#         feature_config_file = self.project_path + "/features/config/feature_config.json"
#         if self.is_raw_data:
#             self.filename = os.path.basename(filename)
#             self.path = os.path.abspath(filename)
#             self.directory = os.path.dirname(self.path)
#             self.category = self.directory.split('/')[-1]
#             self.audio = Audio(self.path, file_type)
#
#         else:
#             self.audio = Audio(is_raw_data=self.is_raw_data, data=buffer)
#
#         with open(feature_config_file) as feature_config:
#             self.feature_config = json.load(feature_config)
#             self.feature_config["sampling_rate"] = self.sampling_rate
#             self.feature_config["frame_size"] = self.frame_size
#             self.feature_config["is_raw_data"] = self.is_raw_data
#
#
#         with self.audio as audio:
#             self.featureManager = FeatureManager()
#
#             self.featureManager.addRegisteredFeatures(FFT(self.audio, self.feature_config), "fft")
#             self.featureManager.addRegisteredFeatures(EMG(self.audio, self.feature_config), "emg")
#
#             self.featureManager.getRegisteredFeature("fft").compute_fft()
#             # self.featureManager.getRegisteredFeature("emg").compute_hurst()
#             self.featureManager.getRegisteredFeature("emg").compute_embed_seq()
#             self.featureManager.getRegisteredFeature("emg").compute_bin_power()
#             # self.featureManager.getRegisteredFeature("emg").compute_pfd()
#             # self.featureManager.getRegisteredFeature("emg").compute_hfd()
#             # self.featureManager.getRegisteredFeature("emg").compute_hjorth()
#             # self.featureManager.getRegisteredFeature("emg").compute_spectral_entropy()
#             # self.featureManager.getRegisteredFeature("emg").compute_svd_entropy()
#             # self.featureManager.getRegisteredFeature("emg").compute_ap_entropy()
#             # self.featureManager.getRegisteredFeature("emg").compute_samp_entropy()
#
#
#             self.feature_list = self.featureManager.getRegisteredFeatures()
#
#     def __repr__(self):
#         return '<{0}/{1}>'.format(self.category, self.filename)
#
#     def get_feature_vector(self):
#         # self.featureManager.getRegisteredFeature("emg").get_hurst()
#         return self.featureManager.getRegisteredFeature("fft").get_logamplitude()
#
# fsamp = 256
# tsample = 1 / fsamp
# f_low = 50
# f_high = 1
# order = 2
# channel_vector = [1,2, 3, 4, 5]
# n_ch = len(channel_vector)
# df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
# df = df[channel_vector].dropna(axis=0)
#
# processed_signal = df.copy()
#
# b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))
#
# b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))
#
# Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
# b3, a3 = butter(order, Wn, btype='stop')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))
#
# project_file_path = "/home/runge/openbci/git/OpenBCI_Python"
# config_file = project_file_path + "/config/config.json"
# with open(config_file) as config:
#     config = json.load(config)
#
#     start = 0
#     end = 400
#     plt.figure(figsize=(12, 8))
#     for h in range(0, n_ch):
#         plt.subplot(n_ch,1,h+1)
#         clip = Clip(config, buffer=np.array(processed_signal.ix[:, h][start * fsamp:end * fsamp].tolist()))
#         # f, Pxx_spec = signal.periodogram(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop',
#         #                                  scaling='spectrum')
#
#         # f, Pxx_spec = signal.welch(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop', 128, scaling='spectrum')
#         # wavelet = signal.ricker
#         # widths = np.arange(1, 11)
#         # cwtmatr = signal.cwt(processed_signal.ix[:, h][start * fsamp:end * fsamp], wavelet, widths)
#         plt.plot(clip.feature_list.get("emg").get_bin_power())
#         # plt.semilogy(fsamp, np.sqrt(Pxx_spec))
#         # plt.ylim([1e-4, 1e1])
#     plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



import json
import os

import seaborn as sb

from features.fft import FFT
from features.generic_type import EMG
from manager import FeatureManager
from py_qt.utils import Audio

sb.set(style="white", palette="muted")

import random
random.seed(20150420)


class Clip:

    def __init__(self, config, buffer=None, filename=None, file_type=None):
        self.is_raw_data = eval(config["is_raw_data"])
        self.frame_size = int(config["window_size"])
        self.sampling_rate = int(config["sampling_rate"])
        # self.project_path = str(config["project_file_path"])
        self.project_path = "/home/runge/openbci/git/OpenBCI_Python"
        feature_config_file = self.project_path + "/features/config/feature_config.json"
        if self.is_raw_data:
            self.filename = os.path.basename(filename)
            self.path = os.path.abspath(filename)
            self.directory = os.path.dirname(self.path)
            self.category = self.directory.split('/')[-1]
            self.audio = Audio(self.path, file_type)

        else:
            self.audio = Audio(is_raw_data=self.is_raw_data, data=buffer)

        with open(feature_config_file) as feature_config:
            self.feature_config = json.load(feature_config)
            self.feature_config["sampling_rate"] = self.sampling_rate
            self.feature_config["frame_size"] = self.frame_size
            self.feature_config["is_raw_data"] = self.is_raw_data


        with self.audio as audio:
            self.featureManager = FeatureManager()

            self.featureManager.addRegisteredFeatures(FFT(self.audio, self.feature_config), "fft")
            self.featureManager.addRegisteredFeatures(EMG(self.audio, self.feature_config), "emg")

            self.featureManager.getRegisteredFeature("fft").compute_fft()
            # self.featureManager.getRegisteredFeature("emg").compute_hurst()
            # self.featureManager.getRegisteredFeature("emg").compute_embed_seq()
            self.featureManager.getRegisteredFeature("emg").compute_bin_power()
            # self.featureManager.getRegisteredFeature("emg").compute_pfd()
            # self.featureManager.getRegisteredFeature("emg").compute_hfd()
            # self.featureManager.getRegisteredFeature("emg").compute_hjorth()
            # self.featureManager.getRegisteredFeature("emg").compute_spectral_entropy()
            # self.featureManager.getRegisteredFeature("emg").compute_svd_entropy()
            # self.featureManager.getRegisteredFeature("emg").compute_ap_entropy()
            # self.featureManager.getRegisteredFeature("emg").compute_samp_entropy()
            self.feature_list = self.featureManager.getRegisteredFeatures()

    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)

    def get_feature_vector(self):
        # self.featureManager.getRegisteredFeature("emg").get_hurst()
        return self.featureManager.getRegisteredFeature("fft").get_logamplitude()

# fsamp = 250
# tsample = 1 / fsamp
# f_low = 50
# f_high = 1
# order = 2
# channel_vector = [1,2, 3, 4, 5]
# n_ch = len(channel_vector)
#
# df1 = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/2noise_signal.csv")
# df1 = df1.dropna(axis=0)
#
# df2 = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/2noise_reduced_signal.csv")
# df2 = df2.dropna(axis=0)
#
# df3 = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/2reconstructed_signal.csv")
# df3 = df3.dropna(axis=0)
#
# df4 = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/2feature_vector.csv")
# df4 = df4.dropna(axis=0)
#
# df = []
# df.append(df1)
# df.append(df2)
# df.append(df3)
# df.append(df4)

# processed_signal = df.copy()
#
# b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))
#
# b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal.ix[:, i]))
#
# Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
# b3, a3 = butter(order, Wn, btype='stop')
# for i in range(0, n_ch):
#     processed_signal.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal.ix[:, i]))
#
# project_file_path = "/home/runge/openbci/git/OpenBCI_Python"
# config_file = project_file_path + "/config/config.json"
# with open(config_file) as config:
#     config = json.load(config)
#
#     start = 100
#     end = 110
#     plt.figure(figsize=(12, 8))
#     for h in range(0, 4):
#         plt.subplot(4,1,h+1)
#         # clip = Clip(config, buffer=np.array(processed_signal.ix[:, h][start * fsamp:end * fsamp].tolist()))
#         # f, Pxx_spec = signal.periodogram(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop',
#         #                                  scaling='spectrum')
#
#         # f, Pxx_spec = signal.welch(processed_signal.ix[:, h][start * fsamp:end * fsamp], fsamp, 'flattop', 128, scaling='spectrum')
#         # wavelet = signal.ricker
#         # widths = np.arange(1, 11)
#         # cwtmatr = signal.cwt(processed_signal.ix[:, h][start * fsamp:end * fsamp], wavelet, widths)
#         plt.plot(df[h].ix[:,0][start * fsamp:end * fsamp])
#         # plt.semilogy(fsamp, np.sqrt(Pxx_spec))
#         # plt.ylim([1e-4, 1e1])
#     plt.show()

































