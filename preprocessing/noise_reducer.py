import json
import random
import socket
import threading
import numpy as np
import tensorflow as tf

from preprocessing import PreProcessor
from RingBuffer import RingBuffer
from utils.dataset_writer_utils import create_sample_from_data
from utils.utils import get_label


class NoiseReducer(threading.Thread):
    def __init__(self, thread_id, input_data, server, lock, writer, config):
        threading.Thread.__init__(self)
        project_path = str(config["project_path"])
        self.window_size = int(config["window_size"])
        self.verbose = eval(config["verbose"])
        self.input_data = input_data
        self.number_of_threads = int(config["number_of_channels"])
        self.feature_vector_size = int(config["feature_vector_size"])
        # self.train_dir = str(config["train_dir_abs_location"])
        self.train_dir = project_path + str(config["train_dir"])
        self.lock = lock
        self.input_buffer = np.zeros([self.number_of_threads, self.window_size])
        self.thread_id = thread_id
        self.output_buffer = []
        self.is_processing = False
        self.server = server
        self.writer = writer
        self.number_of_class = int(config["model"]["number_of_class"])
        self.ip = str(config["ip"])
        self.port = int(config["port"]) + 5  # adding five offset to secondary udp server

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
        self.output_buffer = np.zeros([self.input_buffer.shape[0], self.feature_vector_size])
        threads = []
        lock = threading.Lock()
        thread_list = [i for i in range(0, self.number_of_threads)]
        for thread_id in thread_list:
            thread = PreProcessor(thread_id, self.input_buffer, self.output_buffer, lock)
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
        with open(self.train_dir + "/preprocssed_data.csv", 'a') as f:
            f.write(json.dumps(self.output_buffer.tolist()))
            f.write("\n")

        class_label = get_label(1, self.number_of_class)
        sample = create_sample_from_data(self.output_buffer, class_label)
        self.writer.write(sample.SerializeToString())

        self.send_noise_data(json.dumps(self.input_buffer.tolist()))
        self.send_preprocessed_data(json.dumps(self.output_buffer.tolist()))
        return self.output_buffer

    def send_preprocessed_data(self, data):
        self.server.sendto(data, (self.ip, self.port))

    def send_noise_data(self, data):
        self.server.sendto(data, (self.ip, self.port+1))



project_file_path = "/home/runge/openbci/OpenBCI_Python"
config_file = "/home/runge/openbci/OpenBCI_Python/config/config.json"

with open(config_file) as config:
            plugin_config = json.load(config)
            buffer_size = int(plugin_config["buffer_size"])
            number_of_channels = int(plugin_config["number_of_channels"])
            buffer_capacity = int(plugin_config["buffer_capacity"])
            tfrecords_filename = project_file_path + str(plugin_config["model"]["tfrecords_filename"])
            lock = threading.Lock()
            server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ring_buffers = [RingBuffer(buffer_size * 4) for i in range(0, number_of_channels)]
            for k in range(0, number_of_channels):
                for p in range(0, buffer_size*buffer_capacity):
                    ring_buffers[k].append(random.randint(1,100))
            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            noisereducer_thread =  NoiseReducer("main thread",ring_buffers,server,lock, writer, plugin_config)
            i = 0
            while i<100:
                if not noisereducer_thread.is_processing:
                    print ("------current process-----")
                    noisereducer_thread = NoiseReducer("main thread", ring_buffers,server,lock, writer, plugin_config)
                    noisereducer_thread.start()
                    noisereducer_thread.join()
                    i+=1
            writer.close()



