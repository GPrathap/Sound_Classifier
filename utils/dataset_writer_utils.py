import numpy as np
import tensorflow as tf

from data_types_utils import _int64_feature, _bytes_feature


def create_sample_from_data(clip, clip_label):
    feature_vector = clip.tostring()
    clip_label = clip_label.tostring()
    return tf.train.Example(features=tf.train.Features(feature={
        'clip_height': _int64_feature(clip.shape[1]),
        'clip_width': _int64_feature(clip.shape[0]),
        'clip_raw': _bytes_feature(feature_vector),
        'clip_label_raw': _bytes_feature(clip_label)}))

