import threading

from csv_collect_and_publish import PluginCSVCollectAndPublish
import pandas as pd

from preprocessing.noise_reducer import NoiseReducer


class DataFeeder(threading.Thread):
    def __init__(self, plugin):
        threading.Thread.__init__(self)
        self.is_run = True
        self.plugin = plugin

    def run(self):
        counter = 0
        position = 0
        while self.is_run:
            index_buffer = 0
            for i in range(0, plugin.number_of_channels):
                self.plugin.ring_buffers[index_buffer].append(float(str(df.ix[:, index_buffer][position])))
                index_buffer += 1
                position += 1
                if position == df.shape[0]:
                    self.is_run = False
                    break
                threading._sleep(self.plugin.sampling_time)

project_dir = "/home/runge/openbci/OpenBCI_Python"
plugin = PluginCSVCollectAndPublish()
plugin.activate()
df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
df = df[plugin.channel_vector].dropna(axis=0)
is_run = True
counter = 0
max_iteration = 2000

data_feeder = DataFeeder(plugin)
data_feeder.start()

while is_run:
    if not plugin.noisereducer_thread.is_processing and\
                    plugin.ring_buffers[0].__len__()>plugin.buffer_size:
    # if not plugin.noisereducer_thread.is_processing:
        print ("next process..." + str(plugin.ring_buffers[0].__len__()))
        plugin.noisereducer_thread = NoiseReducer("main thread", plugin.ring_buffers, plugin.server,
                                                  plugin.writer, plugin.plugin_config)
        plugin.noisereducer_thread.start()
        plugin.noisereducer_thread.join()
        counter +=1
        if counter == max_iteration:
            data_feeder.is_run = False
            is_run = False
            data_feeder.join()
    if data_feeder.is_run == False:
        is_run = False

print ("process is done")