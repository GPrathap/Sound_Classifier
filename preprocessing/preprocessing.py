from __future__ import print_function
import Queue
import json
import socket
import sys
import threading
import numpy as np

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
        print("Starting " + str(self.thread_id))
        self.ssa_processor(self.thread_id)
        print("Exiting " + str(self.thread_id))

    def ssa_processor(self, thread_id):
        noise_signal = self.input_buffer[thread_id]
        reconstructed_signal = SingularSpectrumAnalysis(noise_signal, self.window_size).execute()
        self.lock.acquire()
        self.output_buffer[thread_id] = reconstructed_signal
        self.lock.release()

# def process_signal(input_buffer, number_of_threads):
#     output_buffer = np.zeros([input_buffer.shape[0], input_buffer.shape[1]])
#     threads = []
#     lock = threading.Lock()
#     thread_list = [i for i in range(0, number_of_threads)]
#     for thread_id in thread_list:
#         thread = PreProcessor(thread_id, input_buffer, output_buffer, lock)
#         thread.start()
#         threads.append(thread)
#     for t in threads:
#         t.join()
#     return output_buffer


# N = 200
# M = 30
# std_noise = 0
# t = np.array([i for i in range(1, N + 1)]).transpose()
# input_signal = np.zeros([5, N])
# for j in range(0, 5):
#     input_signal[j] = np.array([t[i] * 2 for i in range(0, N)]).transpose()
#
# num_of_threads = 5
# processed_signal = process_signal(input_signal, num_of_threads)
#
# print("prepossessing is done")
