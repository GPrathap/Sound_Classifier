from __future__ import print_function
import Queue
import json
import socket
import sys
import threading

import librosa
import numpy as np

from processor import Clip
from ssa import SingularSpectrumAnalysis


class PreProcessor(threading.Thread):
    def __init__(self, thread_id, input_buffer, output_buffer, lock, window_size=30):
        threading.Thread.__init__(self)
        self.isRun = True
        self.thread_id = thread_id
        self.lock = lock
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.window_size = window_size

    def run(self):
        self.ssa_processor(self.thread_id)

    def ssa_processor(self, thread_id):
        noise_signal = self.input_buffer[thread_id]
        reconstructed_signal = SingularSpectrumAnalysis(noise_signal, self.window_size).execute()
        # clip = Clip(buffer=reconstructed_signal, is_raw_data=False)
        # reconstructed_signal = clip.get_feature_vector()
        self.lock.acquire()
        self.output_buffer[thread_id] = reconstructed_signal
        self.lock.release()
