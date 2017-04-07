import os

import matplotlib

from utils import data_types_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)
import numpy as np
import tensorflow as tf
import time
import librosa
import librosa.display

import numpy
import json
from PIL import Image

from preprocessing.processor import Clip


class DataLoader:
    def __init__(self, project_dir, dataset_dir):

        meta_info_file = project_dir + "/meta_info/configurator.json"

        with open(meta_info_file) as data_file:
            meta_info = json.load(data_file)
            self.conf = meta_info
            # TODO separate out common stuff
            meta_info = meta_info["processing"]["train"]
            self.batch_process_threads_num = int(meta_info["batch_process_threads_num"])
            self.project_dir = project_dir
            self.dataset_dir = dataset_dir
            self.num_epochs = int(meta_info["num_epochs"])
            self.batch_size = int(meta_info["batch_size"])
            self.train_dir = project_dir + str(meta_info["dir"])
            self.train_file = str(meta_info["data_set_name"])
            self.tfrecords_filename = project_dir + str(meta_info["tfrecords_filename"])
            self.number_of_class = int(meta_info["number_of_class"])
            self.generated_image_width = int(meta_info["generated_image_width"])
            self.generated_image_height = int(meta_info["generated_image_height"])
            self.feature_vector_size = self.generated_image_height * self.generated_image_width
            self.num_channels = int(meta_info["num_channels"])
            self.generated_image_dir = project_dir + str(meta_info["generated_image_dir"])
            # TODO add validate and test initialization

    def get_train_config(self):
        return self.conf["processing"]["train"]

    def save_plot_clip_overview(self, clip, i):
        with clip.audio as audio:
            figure = plt.figure(figsize=(self.generated_image_width, self.generated_image_height), dpi=1)
            axis = figure.add_subplot(111)
            plt.axis('off')
            plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                            labeltop='off',
                            labelright='off', labelbottom='off')
            result = np.array(np.array(clip.feature_list['fft'].get_logamplitude()[0:1]))
            librosa.display.specshow(result, sr=clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
            extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
            clip.filename = self.generated_image_dir + clip.filename + str(i) + str("_.jpg")
            plt.savefig(clip.filename, format='jpg', bbox_inches=extent, pad_inches=0, dpi=1)
            plt.close()
        return clip.filename

    def save_clip_overview(self, categories=5, clips_shown=1, clips=None):
        for c in range(0, categories):
            for i in range(0, clips_shown):
                self.save_plot_clip_overview(clips[c][i], i)

    def create_one_big_file(self, file_type):
        writer = tf.python_io.TFRecordWriter(self.tfrecords_filename)
        for directory in sorted(os.listdir('{0}/'.format(self.dataset_dir))):
            store_location = self.generated_image_dir + directory
            # todo make directory if not created
            directory = '{0}/{1}'.format(self.dataset_dir, directory)
            if os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit():
                print('Parsing ' + directory)
                for clip in sorted(os.listdir(directory)):
                    if clip[-3:] == file_type:
                        clip_label, clip_data, rows, _ = self.extracted_sample(directory, clip, file_type)
                        for j in range(0, rows - 2):
                            clip_filename = self.draw_sample_plot_and_save(clip_data, store_location, clip, j)
                            sample = self.create_sample_from_image(clip_filename, clip_label)
                            writer.write(sample.SerializeToString())
                        writer.close()
                        return

        print('All {0} recordings loaded.'.format(self.train_file))

    def extracted_sample(self, directory, clip, file_type):
        print ('{0}/{1}'.format(directory, clip))
        clip_category = ('{0}/{1}'.format(directory, clip), directory.split("/0")[1].
                         split("-")[0].strip())[1]
        clip_data = Clip('{0}/{1}'.format(directory, clip), file_type). \
            get_feature_vector()
        rows = clip_data.shape[0]
        cols = clip_data.shape[1]
        clip_label = self.get_label(int(clip_category)).tostring()
        return clip_label, clip_data, rows, cols

    def draw_sample_plot_and_save(self, raw_data_clip, store_location, clip, index):
        figure = plt.figure(figsize=(
            np.ceil(self.generated_image_width + self.generated_image_width * 0.2),
            np.ceil(self.generated_image_height + self.generated_image_height * 0.2)), dpi=1)
        axis = figure.add_subplot(111)
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',
                        labelleft='off',
                        labeltop='off',
                        labelright='off', labelbottom='off')
        result = np.array(np.array(raw_data_clip[index:index + 1]))
        librosa.display.specshow(result, sr=Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        clip_filename = "%s%s%s%s" % (store_location, clip, str(index), "_.jpg")
        plt.savefig(clip_filename, format='jpg', bbox_inches=extent, pad_inches=0)
        plt.close(figure)
        return clip_filename

    def create_sample_from_image(self, clip_filename, clip_label):
        image = Image.open(clip_filename)
        image = image.resize((self.generated_image_width, self.generated_image_height),
                             Image.ANTIALIAS)
        image = numpy.asarray(image)
        np.reshape(image, (-1, image.shape[1] * image.shape[0]))
        feature_vector = image.tostring()
        clip_raw = tf.train.Example(features=tf.train.Features(feature={
            'clip_height': data_types_utils._int64_feature(image.shape[1]),
            'clip_width': data_types_utils._int64_feature(image.shape[0]),
            'clip_raw': data_types_utils._bytes_feature(feature_vector),
            'clip_label_raw': data_types_utils._bytes_feature(clip_label)}))
        return clip_raw

    def get_label(self, class_number):
        label = np.zeros(self.number_of_class, dtype=np.int)
        label[class_number - 1] = 1
        return label

    def read_tf_recode(self):
        reconstructed_clips = []
        record_iterator = tf.python_io.tf_record_iterator(path=self.tfrecords_filename)
        for string_record in record_iterator:
            raw_clip = tf.train.Example()
            raw_clip.ParseFromString(string_record)
            height = int(raw_clip.features.feature['clip_height'].int64_list.value[0])
            width = int(raw_clip.features.feature['clip_width'].int64_list.value[0])
            img_string = (raw_clip.features.feature['clip_raw'].bytes_list.value[0])
            label = (raw_clip.features.feature['clip_label_raw'].bytes_list.value[0])
            img_1d = np.fromstring(img_string, dtype=np.float64)
            label = np.fromstring(img_string, dtype=np.uint64)
            reconstructed_clip = img_1d.reshape((height, width, -1))
            reconstructed_clip_label = label.reshape((1, self.number_of_class, -1))
            reconstructed_clips.append((reconstructed_clip, reconstructed_clip_label))
        return reconstructed_clips

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'clip_height': tf.FixedLenFeature([], tf.int64),
                'clip_width': tf.FixedLenFeature([], tf.int64),
                'clip_raw': tf.FixedLenFeature([], tf.string),
                'clip_label_raw': tf.FixedLenFeature([], tf.string)
            })
        image = tf.decode_raw(features['clip_raw'], tf.int8)
        label = tf.decode_raw(features['clip_label_raw'], tf.int64)
        image = tf.reshape(image, [self.num_channels, self.feature_vector_size])
        label = tf.reshape(label, [1, self.number_of_class])
        return image, label

    def inputs(self):
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer([self.tfrecords_filename],
                                                            num_epochs=self.num_epochs)
            image, label = self.read_and_decode(filename_queue)
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=self.batch_size, num_threads=self.batch_process_threads_num,
                capacity=1000 + 3 * self.batch_size,
                min_after_dequeue=100)
            return images, sparse_labels

    def run_training(self):
        with tf.Graph().as_default():
            image, label = self.inputs()
            with tf.Session()  as sess:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                try:
                    step = 0
                    while not coord.should_stop():
                        start_time = time.time()
                        while not coord.should_stop():
                            # Run training steps or whatever
                            exatmple, l = sess.run([image, label])
                            print l
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (self.num_epochs, self.batch_size))
                finally:
                    coord.request_stop()
                coord.join(threads)
                sess.close()

# parser = argparse.ArgumentParser(description='Data set location and project location')
# parser.add_argument('-dataset_dir', nargs=2)
# parser.add_argument('-project_dir', nargs=1)
#
# opts = parser.parse_args()
#
# project_dir = opts.project_dir
# dataset_dir = opts.dataset_dir

project_dir = "/home/runge/openbci/OpenBCI_Python"
dataset_dir = "/home/runge/projects/sound_detector/TRAIN-10"

loader = DataLoader(project_dir, dataset_dir)
# # # clips_10 = loader.load_dataset_from_ogg('/home/runge/projects/sound_detector/TRAIN-10')
loader.create_one_big_file("ogg")

# image, label = loader.inputs()
# loader.run_training()
