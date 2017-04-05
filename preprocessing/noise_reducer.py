import json
import socket
import threading
import numpy as np

from preprocessing import PreProcessor
from RingBuffer import RingBuffer


class NoiseReducer(threading.Thread):
    def __init__(self, thread_id, window_size, input_data,number_of_threads, server,ip, port, lock, verbose=False):
        threading.Thread.__init__(self)
        self.window_size = window_size
        self.verbose = verbose
        self.input_data = input_data
        self.lock = lock
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
        if self.verbose:
            print("Starting " + str(self.thread_id))
        self.lock.acquire()
        self.is_processing = True
        self.construct_input_buffer()
        self.process_signal()
        if self.verbose:
            print (self.output_buffer)
        self.is_processing = False
        self.lock.release()
        if self.verbose:
            print("Existing " + str(self.thread_id))

    def process_signal(self):
        self.output_buffer = np.zeros([self.input_buffer.shape[0], int(np.ceil(self.window_size))])
        threads = []
        lock = threading.Lock()
        thread_list = [i for i in range(0, self.number_of_threads)]
        for thread_id in thread_list:
            thread = PreProcessor(thread_id, self.input_buffer, self.output_buffer, lock)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        with open("preprocssed_data.csv", 'a') as f:
            f.write(json.dumps(self.output_buffer.tolist()))
            f.write("\n")
        self.send_noise_data(json.dumps(self.input_buffer.tolist()))
        self.send_preprocessed_data(json.dumps(self.output_buffer.tolist()))
        return self.output_buffer

    def send_preprocessed_data(self, data):
        self.server.sendto(data, (self.ip, self.port))

    def send_noise_data(self, data):
        self.server.sendto(data, (self.ip, self.port+1))


# buffer_size = 60
# number_of_channels = 8
# ip = "0.0.0.0"
# port = 8893
# lock = threading.Lock()
# server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ring_buffers = [RingBuffer(buffer_size*4) for i in range(0, number_of_channels)]
# noisereducer_thread =  NoiseReducer("main thread", buffer_size, ring_buffers,number_of_channels, server, ip, port, lock)
#
# i = 0
# while i<100:
#     if not noisereducer_thread.is_processing:
#         print "------current process-----"
#         noisereducer_thread = NoiseReducer("main thread", buffer_size, ring_buffers, number_of_channels, server, ip, port, lock)
#         noisereducer_thread.start()
#         noisereducer_thread.join()
#         threading._sleep(2)
#
#         i+=1



