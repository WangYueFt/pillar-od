"""Waymo Data loader."""

import functools
import os
import numpy as np

import tf_util
from data import waymo_decoder

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


def augment(points_xyz, points_mask, bboxes):
  """data augmentation."""
  rand = tf.random.uniform([],
                           minval=-1.0,
                           maxval=1.0,
                           dtype=tf.dtypes.float32)
  rand = tf.where(rand > 0, 1, -1)
  rand = tf.cast(rand, tf.dtypes.float32)
  points_xyz = tf.concat([points_xyz[:, 0:1],
                          points_xyz[:, 1:2] * rand,
                          points_xyz[:, 2:]],
                         axis=-1)
  bboxes = tf.concat([bboxes[:, 0:1],
                      bboxes[:, 1:2] * rand,
                      bboxes[:, 2:6],
                      bboxes[:, 6:] * rand],
                     axis=-1)
  theta = tf.random.uniform([],
                            minval=-1,
                            maxval=1,
                            dtype=tf.dtypes.float32) * np.pi / 4.0
  rz = tf.stack([tf.cos(theta), tf.sin(theta), 0,
                 -tf.sin(theta), tf.cos(theta), 0,
                 0, 0, 1])
  rz = tf.reshape(rz, [3, 3])
  points_xyz = tf.matmul(points_xyz, rz)
  theta = tf.reshape(theta, [1, 1])
  bboxes = tf.concat(
      [tf.matmul(bboxes[:, 0:3], rz),
       bboxes[:, 3:6],
       tf_util.wrap_angle_rad(bboxes[:, 6:] + theta, -np.pi, np.pi)], axis=-1)
  jitter = tf.random.normal(points_xyz.shape, 0.0, 0.02)
  points_xyz = points_xyz + jitter
  return points_xyz, points_mask, bboxes


def add_points_bboxes(points_xyz, bboxes, bboxes_label, bboxes_mask,
                      is_2d=False):
  """Assign bboxes to points."""
  one_bbox = bboxes[-1:, :]
  one_bbox_mask = tf.zeros([1])
  bboxes = bboxes[bboxes_mask == 1]
  bboxes_mask = bboxes_mask[bboxes_mask == 1]

  bboxes = tf.concat([bboxes, one_bbox], axis=0)
  bboxes_mask = tf.concat([bboxes_mask, one_bbox_mask], axis=0)

  theta = bboxes[:, 6]
  rz = tf.stack(
      [tf.cos(theta), -tf.sin(theta), tf.zeros_like(theta),
       tf.sin(theta), tf.cos(theta), tf.zeros_like(theta),
       tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)],
      axis=-1)
  rz = tf.reshape(rz, [-1, 3, 3])

  points_xyz = tf.reshape(points_xyz, [-1, 1, 3])

  points_xyz_in_bbox_frame = (tf.reshape(points_xyz, [-1, 1, 3])-
                              tf.reshape(bboxes[:, 0:3], [1, -1, 3]))
  # points_xyz -> (n, m, 1, 3) * (1, m, 3, 3) -> (n, m, 1, 3)
  points_xyz_in_bbox_frame = tf.expand_dims(points_xyz_in_bbox_frame, axis=2)
  rz = tf.expand_dims(rz, axis=0)

  # (n, m, 3)
  points_xyz_in_bbox_frame = tf.squeeze(tf.matmul(points_xyz_in_bbox_frame, rz),
                                        axis=2)

  # (1, m, 3)
  bboxes_size_min = tf.expand_dims(-bboxes[:, 3:6] / 2, axis=0)
  bboxes_size_max = tf.expand_dims(bboxes[:, 3:6] / 2, axis=0)

  # (n, m)
  if is_2d:
    points_if_in_bboxes = tf.reduce_all(
        (points_xyz_in_bbox_frame[:, :, 0:2] >= bboxes_size_min[:, :, 0:2]) &
        (points_xyz_in_bbox_frame[:, :, 0:2] <= bboxes_size_max[:, :, 0:2]),
        axis=-1)

    points_centerness_left = tf.math.abs(
        points_xyz_in_bbox_frame[:, :, 0:2] - bboxes_size_min[:, :, 0:2])
    points_centerness_right = tf.math.abs(
        bboxes_size_max[:, :, 0:2] - points_xyz_in_bbox_frame[:, :, 0:2])
    points_centerness_min = tf.math.minimum(points_centerness_left,
                                            points_centerness_right)
    points_centerness_max = tf.math.maximum(points_centerness_left,
                                            points_centerness_right)

    points_centerness = tf.math.sqrt(
        tf.math.reduce_prod(points_centerness_min / points_centerness_max,
                            axis=-1))

  else:
    points_if_in_bboxes = tf.reduce_all(
        (points_xyz_in_bbox_frame >= bboxes_size_min) &
        (points_xyz_in_bbox_frame <= bboxes_size_max), axis=-1)

    points_centerness_left = tf.math.abs(
        points_xyz_in_bbox_frame - bboxes_size_min)
    points_centerness_right = tf.math.abs(
        bboxes_size_max - points_xyz_in_bbox_frame)
    points_centerness_min = tf.math.minimum(points_centerness_left,
                                            points_centerness_right)
    points_centerness_max = tf.math.maximum(points_centerness_left,
                                            points_centerness_right)
    # (N, M)
    points_centerness = tf.math.sqrt(
        tf.math.reduce_prod(points_centerness_min / points_centerness_max,
                            axis=-1))

  points_if_in_bboxes = tf.cast(points_if_in_bboxes, tf.dtypes.float32)

  # n, m
  points_if_in_bboxes = points_if_in_bboxes * tf.reshape(bboxes_mask, [1, -1])

  # n
  points_bboxes_index = tf.argmax(points_if_in_bboxes, axis=1)

  points_if_in_bboxes = tf.gather(points_if_in_bboxes,
                                  points_bboxes_index, batch_dims=1)
  points_centerness = tf.gather(points_centerness,
                                points_bboxes_index, batch_dims=1)

  points_bboxes = tf.gather(bboxes, points_bboxes_index, axis=0)
  points_bboxes_label = tf.gather(bboxes_label, points_bboxes_index, axis=0)

  return (points_bboxes, points_bboxes_label,
          points_if_in_bboxes, points_centerness, points_bboxes_index)


def assign_bboxes(pillar_map_size=(256, 256),
                  pillar_map_range=(75.2, 75.2),
                  bboxes=None,
                  bboxes_mask=None,
                  bboxes_label=None,):
  """Assign bboxes to birds-eye view pillars."""
  half_size_height = pillar_map_range[0] * 2 / pillar_map_size[0] / 2
  half_size_width = pillar_map_range[1] * 2 / pillar_map_size[1] / 2
  height_range = tf.linspace(-pillar_map_range[0] + half_size_height,
                             pillar_map_range[0] - half_size_height,
                             pillar_map_size[0])
  width_range = tf.linspace(-pillar_map_range[1] + half_size_width,
                            pillar_map_range[1] - half_size_width,
                            pillar_map_size[1])
  height_range = tf.reshape(height_range, [pillar_map_size[0], 1])
  width_range = tf.reshape(width_range, [1, pillar_map_size[1]])
  height_range = tf.tile(height_range, [1, pillar_map_size[1]])
  width_range = tf.tile(width_range, [pillar_map_size[0], 1])
  z_range = tf.zeros_like(height_range)
  pillar_map_xyz = tf.stack([height_range, width_range, z_range], axis=2)
  pillar_map_xyz = tf.reshape(pillar_map_xyz, [-1, 3])
  (pillar_map_bboxes, pillar_map_bboxes_label, pillar_map_if_in_bboxes,
   pillar_map_centerness, pillar_map_bboxes_index) = add_points_bboxes(
       pillar_map_xyz, bboxes, bboxes_label, bboxes_mask, is_2d=True)
  pillar_map_xyz = tf.reshape(pillar_map_xyz,
                              [pillar_map_size[0], pillar_map_size[1], 3])
  pillar_map_bboxes = tf.reshape(pillar_map_bboxes,
                                 [pillar_map_size[0], pillar_map_size[1], 7])
  pillar_map_bboxes_label = tf.reshape(
      pillar_map_bboxes_label, [pillar_map_size[0], pillar_map_size[1]])
  pillar_map_if_in_bboxes = tf.reshape(
      pillar_map_if_in_bboxes, [pillar_map_size[0], pillar_map_size[1]])
  pillar_map_bboxes_index = tf.reshape(
      pillar_map_bboxes_index, [pillar_map_size[0], pillar_map_size[1]])
  return (pillar_map_xyz, pillar_map_bboxes, pillar_map_bboxes_label,
          pillar_map_if_in_bboxes, pillar_map_centerness,
          pillar_map_bboxes_index)


def decode_fn(value, data_aug=False,
              max_num_points=245760, max_num_bboxes=100,
              class_id=1, difficulty=1, pillar_map_size=(256, 256),
              pillar_map_range=(75.2, 75.2)):
  """Decode function."""

  tensor_dict = waymo_decoder.decode_tf_example(
      serialized_example=value,
      features=waymo_decoder.FEATURE_SPEC)

  frame_name = tensor_dict['frame_name']
  points_xyz = tensor_dict['lidars']['points_xyz']
  points_feature = tensor_dict['lidars']['points_feature']

  bboxes = tensor_dict['objects']['box']
  bboxes_label = tensor_dict['objects']['label']
  bboxes_speed = tensor_dict['objects']['speed']
  bboxes_difficulty = tensor_dict['objects']['combined_difficulty_level']

  num_valid_points = tf_util.get_shape(points_xyz)[0]
  points_xyz = tf_util.pad_or_trim_to(points_xyz, [max_num_points, 3])
  points_feature = tf_util.pad_or_trim_to(points_feature, [max_num_points, 2])

  points_mask = tf.sequence_mask(num_valid_points,
                                 maxlen=max_num_points)

  points_mask = tf.cast(points_mask, dtype=tf.dtypes.float32)

  bboxes_difficulty = bboxes_difficulty <= difficulty
  bboxes_mask = tf.equal(bboxes_label, class_id)
  bboxes_mask = tf.math.logical_and(bboxes_mask, bboxes_difficulty)
  bboxes_mask = tf.cast(bboxes_mask, dtype=tf.dtypes.float32)

  num_valid_bboxes = tf_util.get_shape(bboxes)[0]
  bboxes_index = tf.math.top_k(
      bboxes_mask, k=tf.math.minimum(max_num_bboxes, num_valid_bboxes))[1]
  bboxes_mask = tf.gather(bboxes_mask, bboxes_index)
  bboxes_label = tf.gather(bboxes_label, bboxes_index)
  bboxes = tf.gather(bboxes, bboxes_index)
  bboxes_speed = tf.gather(bboxes_speed, bboxes_index)

  bboxes = tf_util.pad_or_trim_to(bboxes, [max_num_bboxes, 7])
  bboxes_label = tf_util.pad_or_trim_to(bboxes_label, [max_num_bboxes])
  bboxes_speed = tf_util.pad_or_trim_to(bboxes_speed, [max_num_bboxes, 2])
  bboxes_difficulty = tf_util.pad_or_trim_to(bboxes_difficulty,
                                             [max_num_bboxes])
  bboxes_mask = tf_util.pad_or_trim_to(bboxes_mask, [max_num_bboxes])

  if data_aug:
    points_xyz, points_mask, bboxes = augment(

        points_xyz=points_xyz,
        points_mask=points_mask,
        bboxes=bboxes)

  (pillar_map_xyz, pillar_map_bboxes, pillar_map_bboxes_label,
   pillar_map_if_in_bboxes, pillar_map_centerness, pillar_map_bboxes_index) = (
       assign_bboxes(
           pillar_map_size=pillar_map_size,
           pillar_map_range=pillar_map_range,
           bboxes=bboxes,
           bboxes_label=bboxes_label,
           bboxes_mask=bboxes_mask))

  pillar_map_xyz = tf.reshape(pillar_map_xyz, [-1, 3])
  pillar_map_bboxes = tf.reshape(pillar_map_bboxes, [-1, 7])
  pillar_map_bboxes_label = tf.reshape(pillar_map_bboxes_label, [-1])
  pillar_map_if_in_bboxes = tf.reshape(pillar_map_if_in_bboxes, [-1])
  pillar_map_centerness = tf.reshape(pillar_map_centerness, [-1])
  pillar_map_bboxes_index = tf.reshape(pillar_map_bboxes_index, [-1])

  return {
      'frame_name': frame_name,
      'points_xyz': points_xyz,
      'points_feature': points_feature,
      'points_mask': points_mask,
      'bboxes': bboxes,
      'bboxes_label': bboxes_label,
      'bboxes_mask': bboxes_mask,
      'bboxes_speed': bboxes_speed,
      'pillar_map_xyz': pillar_map_xyz,
      'pillar_map_bboxes': pillar_map_bboxes,
      'pillar_map_if_in_bboxes': pillar_map_if_in_bboxes,
  }


def waymo_open_dataset(batch_size=64,
                       data_path=None,
                       split='train',
                       cycle_length=256,
                       shuffle_buffer_size=4096,
                       num_parallel_calls=256,
                       percentile=1.0,
                       max_num_points=245760,
                       max_num_bboxes=100,
                       class_id=1,
                       difficulty=1,
                       pillar_map_size=(256, 256),
                       pillar_map_range=(75.2, 75.2)):
  """Waymo open dataset loader."""

  file_pattern = os.path.join(data_path, '%s-*' % split)
  is_train = split == 'train'
  files = tf.data.Dataset.list_files(
      file_pattern, shuffle=False)
  if not is_train and percentile < 1.0:
    files = files.take(int(percentile*5720))
  elif is_train:
    files = files.shuffle(1000)

  if num_parallel_calls == -1:
    num_parallel_calls = tf.data.experimental.AUTOTUNE
  dataset = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=cycle_length,
      num_parallel_calls=num_parallel_calls)
  dataset = dataset.map(functools.partial(decode_fn,
                                          data_aug=is_train,
                                          max_num_points=max_num_points,
                                          max_num_bboxes=max_num_bboxes,
                                          class_id=class_id,
                                          difficulty=difficulty,
                                          pillar_map_size=pillar_map_size,
                                          pillar_map_range=pillar_map_range),
                        num_parallel_calls=num_parallel_calls)
  if is_train:
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size, drop_remainder=is_train)
  dataset = dataset.prefetch(buffer_size=1)
  return dataset

