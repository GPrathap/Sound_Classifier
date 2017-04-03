import json
import socket
import threading
import numpy as np

from preprocessing import PreProcessor


class NoiseReducer(threading.Thread):
    def __init__(self, thread_id, window_size, input_data,number_of_threads, server,ip, port, verbose=False):
        threading.Thread.__init__(self)
        self.window_size = window_size
        self.verbose = verbose
        self.input_data = input_data
        self.input_buffer = np.zeros([number_of_threads, window_size])
        self.thread_id = thread_id
        self.number_of_threads = number_of_threads
        self.output_buffer = []
        self.is_processing = False
        self.server = server
        self.ip = ip
        self.port = port

    def construct_input_buffer(self):
        for j in range(0, len(self.input_data)):
            try:
                self.input_buffer[j] = self.input_data[j].pop_window(self.window_size)
            except:
                self.input_buffer[j] = [i for i in range(0, self.window_size)]
                pass
        self.input_buffer = np.array(self.input_buffer)

    def run(self):
        print("Starting " + str(self.thread_id))
        self.is_processing = True
        self.construct_input_buffer()
        self.process_signal()
        self.is_processing = False
        print("Exiting " + str(self.thread_id))

    def process_signal(self):
        self.output_buffer = np.zeros([self.input_buffer.shape[0], self.input_buffer.shape[1]])
        threads = []
        lock = threading.Lock()
        thread_list = [i for i in range(0, self.number_of_threads)]
        for thread_id in thread_list:
            thread = PreProcessor(thread_id, self.input_buffer, self.output_buffer, lock)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        self.send_preprocessed_data(json.dumps(self.output_buffer[0].tolist()))
        return self.output_buffer

    def send_preprocessed_data(self, data):
        self.server.sendto(data, (self.ip, self.port))


# buffer_size = 1024
# number_of_channels = 8
# ip = "0.0.0.0"
# port = 8889
# server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ring_buffers = [RingBuffer(buffer_size*4) for i in range(0, number_of_channels)]
# noisereducer_thread =  NoiseReducer("main thread", 1024, ring_buffers,number_of_channels, server, ip, port)
# noisereducer_thread.start()
# noisereducer_thread.join()
# print noisereducer_thread.output_buffer


