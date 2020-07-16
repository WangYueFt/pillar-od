"""Waymo open dataset decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zlib
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
tf.enable_v2_behavior()


OBJECT_SPEC = tfds.features.FeaturesDict({
    'id': tf.int64,
    'name': tfds.features.Text(),
    'num_points': tf.int64,
    # detection difficulty level
    'detection_difficulty_level': tf.int64,
    # combined difficulty level :
    # detection_difficulty_level & (num_points < 5)
    'combined_difficulty_level': tf.int64,
    'speed': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
    'accel': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
    # object category (class).
    'label': tf.int64,
    # box parameters
    'box': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
})


# Feature specification of waymo open dataset.
FEATURE_SPEC = tfds.features.FeaturesDict({
    'scene_name': tfds.features.Text(),
    'frame_name': tfds.features.Text(),
    'lidars': {
        'points_xyz':
            tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
        'points_feature':
            tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
    },
    'objects': tfds.features.Sequence(OBJECT_SPEC),
})


def decode_tf_example(serialized_example, features):
  """Decodes a serialized Example proto as dictionary of tensorflow tensors."""
  example_specs = features.get_serialized_info()
  parser = tfds.core.example_parser.ExampleParser(example_specs)
  tfexample_data = parser.parse_example(serialized_example)
  if isinstance(features, tfds.features.FeaturesDict):
    features._set_top_level()  # pylint: disable=protected-access
  return features.decode_example(tfexample_data)


def encode_tf_example(example_data, features):
  """Encode the feature dict into a tf.train.Eexample proto string."""
  encoded = features.encode_example(example_data)
  example_specs = features.get_serialized_info()
  serializer = tfds.core.example_serializer.ExampleSerializer(example_specs)
  return serializer.serialize_example(encoded)


def decode_frame(frame):
  """Decodes native waymo Frame proto to tf.Examples."""

  lidars = extract_points(frame.lasers,
                          frame.context.laser_calibrations,
                          frame.pose)
  objects = extract_objects(frame.laser_labels)
  frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
      scene_name=frame.context.name,
      location=frame.context.stats.location,
      time_of_day=frame.context.stats.time_of_day,
      timestamp=frame.timestamp_micros)

  example_data = {
      'scene_name': frame.context.name,
      'frame_name': frame_name,
      'lidars': lidars,
      'objects': objects,
  }
  return encode_tf_example(example_data, FEATURE_SPEC)


def extract_points_from_range_image(laser, calibration, frame_pose):
  """Decode points from lidar."""
  if laser.name != calibration.name:
    raise ValueError('Laser and calibration do not match')
  if laser.name == dataset_pb2.LaserName.TOP:
    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame_pose.transform), [4, 4]))
    range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
        zlib.decompress(laser.ri_return1.range_image_pose_compressed))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[...,
                                                                         2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[...,
                                                                          3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    frame_pose = tf.expand_dims(frame_pose, axis=0)
    pixel_pose = tf.expand_dims(range_image_top_pose_tensor, axis=0)
  else:
    pixel_pose = None
    frame_pose = None
  first_return = zlib.decompress(
      laser.ri_return1.range_image_compressed)
  second_return = zlib.decompress(
      laser.ri_return2.range_image_compressed)
  points_list = []
  for range_image_str in [first_return, second_return]:
    range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
    if not calibration.beam_inclinations:
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([
              calibration.beam_inclination_min, calibration.beam_inclination_max
          ]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(calibration.beam_inclinations)
    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(calibration.extrinsic.transform), [4, 4])
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose))
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(
        tf.concat([range_image_cartesian, range_image_tensor[..., 1:4]],
                  axis=-1),
        tf.where(range_image_mask))
    points_list.append(points_tensor.numpy())
  return points_list


def extract_points(lasers, laser_calibrations, frame_pose):
  """Extract point clouds."""
  sort_lambda = lambda x: x.name
  lasers_with_calibration = zip(
      sorted(lasers, key=sort_lambda),
      sorted(laser_calibrations, key=sort_lambda))
  points_xyz = []
  points_feature = []
  points_nlz = []
  for laser, calibration in lasers_with_calibration:
    points_list = extract_points_from_range_image(laser, calibration,
                                                  frame_pose)
    points = np.concatenate(points_list, axis=0)
    points_xyz.extend(points[..., :3].astype(np.float32))
    points_feature.extend(points[..., 3:5].astype(np.float32))
    points_nlz.extend(points[..., 5].astype(np.float32))
  return {
      'points_xyz': np.asarray(points_xyz),
      'points_feature': np.asarray(points_feature),
  }


def extract_objects(laser_labels):
  """Extract objects."""
  objects = []
  for object_id, label in enumerate(laser_labels):
    category_label = label.type
    box = label.box

    speed = [label.metadata.speed_x, label.metadata.speed_y]
    accel = [label.metadata.accel_x, label.metadata.accel_y]
    num_lidar_points_in_box = label.num_lidar_points_in_box
    # Difficulty level is 0 if labeler did not say this was LEVEL_2.
    # Set difficulty level of "999" for boxes with no points in box.
    if num_lidar_points_in_box <= 0:
      combined_difficulty_level = 999
    if label.detection_difficulty_level == 0:
      # Use points in box to compute difficulty level.
      if num_lidar_points_in_box >= 5:
        combined_difficulty_level = 1
      else:
        combined_difficulty_level = 2
    else:
      combined_difficulty_level = label.detection_difficulty_level

    objects.append({
        'id': object_id,
        'name': label.id,
        'label': category_label,
        'box': np.array([box.center_x, box.center_y, box.center_z,
                         box.length, box.width, box.height, box.heading],
                        dtype=np.float32),
        'num_points':
            num_lidar_points_in_box,
        'detection_difficulty_level':
            label.detection_difficulty_level,
        'combined_difficulty_level':
            combined_difficulty_level,
        'speed':
            np.array(speed, dtype=np.float32),
        'accel':
            np.array(accel, dtype=np.float32),
    })
  return objects
