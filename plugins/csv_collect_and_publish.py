import Queue
import csv
import json
import socket
import threading
import timeit
import datetime

from preprocessing.RingBuffer import RingBuffer
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.server import UDPServer

import plugin_interface as plugintypes


class PluginCSVCollectAndPublish(plugintypes.IPluginExtended):
    def __init__(self, file_name="collect.csv", ip='0.0.0.0', port=8888, secondary_port=8889, test=True,
                 receiver_port=4096, delim=",", verbose=False, train=True, number_of_channels =5, buffer_size=64):
        now = datetime.datetime.now()
        self.time_stamp = '%d-%d-%d_%d-%d-%d' % (now.year, now.month, now.day, now.hour,
                                                 now.minute, now.second)
        self.train = train
        self.file_name = self.time_stamp
        self.start_time = timeit.default_timer()
        self.delim = delim
        self.verbose = verbose
        self.test = test
        self.ip = ip
        self.port = port
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.queue = Queue.Queue(5)
        self.number_of_channels = number_of_channels
        self.buffer_size = buffer_size*1
        self.ring_buffers = [RingBuffer(self.buffer_size) for i in range(0, self.number_of_channels)]
        self.hh = 1
        self.noisereducer_thread = NoiseReducer("main thread", self.buffer_size,
                                              self.ring_buffers,number_of_channels, self.server, self.ip, self.port+1, self.lock, self.verbose)
        if self.train:
            self.secondary_ip = ip
            self.secondary_port = secondary_port
            self.receiver_port = receiver_port
            self.secondary_server = UDPServer("udp_server", self.queue, self.secondary_port
                                              , self.receiver_port, ip=self.secondary_ip)

    def activate(self):
        if len(self.args) > 0:
            if 'no_time' in self.args:
                self.file_name = self.args[0]
            else:
                self.file_name = self.args[0] + '_' + self.file_name
            if 'verbose' in self.args:
                self.verbose = True

        self.file_name = self.file_name + '.csv'
        print ("Will export CSV to:", self.file_name)

        print ("udp_server plugin")
        print (self.args)

        if len(self.args) > 0:
            self.ip = self.args[0]
        if len(self.args) > 1:
            self.port = int(self.args[1])

        # init network
        print ("Selecting raw UDP streaming. IP: ", self.ip, ", port: ", str(self.port))
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print ("Server started on port " + str(self.port))
        # init kinect network
        if self.train:
            try:
                self.secondary_server.start()
                # self.secondary_server.isRun = False
                # self.secondary_server.join()
                print ("Selecting raw UDP streaming. IP: ", self.secondary_ip, ", port: ",\
                    str(self.secondary_port))
            except :
                print "Error while starting udp server..."
                self.secondary_server.socket.close()

        # Open in append mode
        with open(self.file_name, 'a') as f:
            f.write('%' + self.time_stamp + '\n')

    def deactivate(self):
        print ("Closing, CSV saved to:", self.file_name)
        self.server.close()
        return

    def show_help(self):
        print ("Optional argument: [filename] (default: collect.csv)")
        print ("""Optional arguments: [ip [port]]
              \t ip: target IP address (default: 'localhost')
              \t port: target port (default: 12345)""")

    def send_row_data(self, data):
        self.server.sendto(data, (self.ip, self.port))

    def get_updated_values(self):
        return self.secondary_server.get_next_point()

    def __call__(self, sample):
        t = timeit.default_timer() - self.start_time

        # print timeSinceStart|Sample Id
        if self.verbose:
            print("CSV: %f | %d" % (t, sample.id))

        kinect_angles = self.get_updated_values()

        if self.train:
            for i in kinect_angles:
                sample.channel_data.append(str(i))

        self.send_row_data(json.dumps(sample.channel_data))

        row = ''
        row += str(t)
        row += self.delim
        row += str(sample.id)
        row += self.delim
        index_buffer =0
        for i in sample.channel_data:
            if self.test:
                if not (index_buffer >= self.number_of_channels):
                    self.ring_buffers[index_buffer].append(float(str(i)))
            row += str(i)
            row += self.delim
            index_buffer+=1
        if self.train:
            for i in kinect_angles:
                row += str(i)
                row += self.delim
                if self.verbose:
                    print (kinect_angles)
                self.hh+=1
        row[-1].replace(",", "")
        row += '\n'
        with open(self.file_name, 'a') as f:
            f.write(row)

        if not self.noisereducer_thread.is_processing:
             self.noisereducer_thread = NoiseReducer("main thread", self.buffer_size, self.ring_buffers,
                                                self.number_of_channels, self.server, self.ip, self.port+5, self.lock, self.verbose)
             self.noisereducer_thread.start()
             self.noisereducer_thread.join()



