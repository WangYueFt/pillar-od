"""Tool to convert Waymo Open Dataset to tf.Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import glob

import time
import tensorflow.compat.v2 as tf
import waymo_decoder
from waymo_open_dataset import dataset_pb2

tf.enable_v2_behavior()

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_filebase', None, 'Path to write output')

FLAGS = flags.FLAGS


def main(unused_argv):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  assert FLAGS.input_file_pattern
  assert FLAGS.output_filebase

  for idx, fname in enumerate(list(glob.glob(FLAGS.input_file_pattern))):
    t1 = time.time()
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    with tf.io.TFRecordWriter(FLAGS.output_filebase + '-%d' % idx) as writer:
      for data in dataset:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = waymo_decoder.decode_frame(frame)
        writer.write(decoded_frame)
    t2 = time.time()
    print('idx:', idx, 'time:', t2 - t1, 'filename:', fname)

if __name__ == '__main__':
  app.run(main)
