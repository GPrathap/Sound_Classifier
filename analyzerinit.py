from __future__ import print_function

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=20)
# manager = plt.get_current_fig_manager()
# manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import manhattan_distances
from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis


class SignalAnalyzer():
    def __init__(self, activity_type, project_path, dataset_location):
        self.raw_data = pd.read_csv(dataset_location)
        self.config_file = project_path + "/config/config.json"
        self.raw_data = self.raw_data.ix[:, 0:4].dropna()
        self.raw_channel_data = self.raw_data
        self.channel_length = self.raw_channel_data.shape[1]
        # self.signal_types = ["noise_signal", "noise_reduced_signal", "feature_vector"]
        self.signal_types = ["noise_signal"]
        self.raw_channel_data_set = []
        self.output_buffer = []
        self.activity_type = activity_type
        self.project_path = project_path
        self.dataset_location = dataset_location
        self.channels_names = ["ch1", "ch2", "ch3", "ch4"]
        with open(self.config_file) as config:
            self.config = json.load(config)
            self.config["train_dir_abs_location"] = self.project_path + "/build/dataset/train"

    def nomalize_signal(self, input_signal):
        mean = np.mean(input_signal, axis=0)
        input_signal -= mean
        return input_signal / np.std(input_signal, axis=0)

    def reconstructed_channel_data(self):
        for i in range(0, self.channel_length):
            self.raw_channel_data_set.append(self.nomalize_signal(self.raw_channel_data.ix[:, i]))
        for i in range(0, self.channel_length):
            preprocessor = PreProcessor(i, self.raw_channel_data_set, self.output_buffer, self.config)
            preprocessor.processor(i, activity_type=activity_type)

    def append_channel_data(self):
        for i in range(0, len(self.signal_types)):
            signal_type = self.signal_types[i]
            noise_signals = []
            for i in range(0, self.channel_length):
                processed_signal = pd.read_csv(str(self.config["train_dir_abs_location"]) + "/" + str(i) + "_" +
                                               activity_type + "_" + signal_type + ".csv")
                noise_signals.append(np.array(processed_signal.ix[:, 0]))
            np.savetxt(str(self.config["train_dir_abs_location"]) + "/result/" + activity_type + "_" +
                       signal_type + "s" + ".csv", np.transpose(np.array(noise_signals)), delimiter=',')

    def execute(self, is_init=False):
        if is_init:
            self.reconstructed_channel_data()
            self.append_channel_data()


project_path = "/home/geesara/project/OpenBCI_Python"
dataset_location = project_path+ "/build/dataset/inittryout.csv"
activity_type = "straight_up"


signal_analyzer = SignalAnalyzer(activity_type, project_path, dataset_location)
signal_analyzer.execute(is_init=True)
