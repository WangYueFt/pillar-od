"""Utitlities."""
import functools

from lingvo.tasks.car import detection_decoder
from lingvo.tasks.car.waymo import waymo_metadata
import numpy as np
import tensorflow.compat.v2 as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2
from waymo_open_dataset.protos import metrics_pb2
tf.enable_v2_behavior()


def pad_or_trim_to(x, shape, pad_val=0):
  """Pad and slice x to the given shape.

  Args:
    x: A tensor.
    shape: The shape of the returned tensor.
    pad_val: An int or float used to pad x.

  Returns:
    'x' is padded with pad_val and sliced so that the result has the given
    shape.
  """
  pad = shape - tf.minimum(tf.shape(x), shape)
  zeros = tf.zeros_like(pad)
  x = tf.pad(x, tf.stack([zeros, pad], axis=1), constant_values=pad_val)
  # If dim-i is larger than shape[i], we slice [0:shape[i]] for dim-i.
  return tf.reshape(tf.slice(x, zeros, shape), shape)


def get_shape(tensor, ndims=None):
  """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.

  Tries to return static shape if it's available. Note that this means
  some of the outputs will be ints while the rest will be Tensors.

  Args:
    tensor: The input tensor.
    ndims: If not None, returns the shapes for the first `ndims` dimensions.
  """
  tensor = tf.convert_to_tensor(tensor)
  dynamic_shape = tf.shape(tensor)

  # Early exit for unranked tensor.
  if tensor.shape.ndims is None:
    if ndims is None:
      return dynamic_shape
    else:
      return [dynamic_shape[x] for x in range(ndims)]

  # Ranked tensor.
  if ndims is None:
    ndims = tensor.shape.ndims
  else:
    ndims = min(ndims, tensor.shape.ndims)

  # Return mixture of static and dynamic dims.
  static_shape = tensor.shape.as_list()
  shapes = [
      static_shape[x] if static_shape[x] is not None else dynamic_shape[x]
      for x in range(ndims)
  ]
  return shapes


def raval_index(coords, dims):
  multiplier = tf.math.cumprod(dims, exclusive=True, reverse=True)
  indices = tf.reduce_sum(coords * multiplier, axis=1)
  return indices


def _batched_unsorted_segment_fn(batched_data,
                                 batched_segment_ids,
                                 num_segments,
                                 unsorted_segment_fn,
                                 batched_padding=None,
                                 name=None):
  """Calls an unsorted segment function on a batch of data."""
  batch_size = get_shape(batched_data)[0]
  batched_segment_shape = get_shape(batched_segment_ids)

  # Convert segment_id -> batch_idx * num_segments + segment_id so that each
  # batch is placed in a different range of segment ids.
  segment_id_start = tf.range(0, batch_size, dtype=batched_segment_ids.dtype)
  segment_id_start *= num_segments

  # Broadcast and add.
  segment_id_start = tf.reshape(segment_id_start,
                                [-1] + [1] * (len(batched_segment_shape) - 1))
  batched_segment_ids += segment_id_start

  # Padded elements should have segment_ids set to -1, so that they are ignored.
  if batched_padding is not None:
    batched_segment_ids = tf.where(
        tf.equal(batched_padding, 1.0),
        -tf.ones_like(batched_segment_ids, dtype=batched_segment_ids.dtype),
        batched_segment_ids)

  batched_segment_output = unsorted_segment_fn(batched_data,
                                               batched_segment_ids,
                                               batch_size * num_segments, name)

  output_shape = get_shape(batched_segment_output)

  # Reshape to recover batch dimension.
  batched_segment_output = tf.reshape(
      batched_segment_output, [batch_size, num_segments] + output_shape[1:])

  return batched_segment_output


batched_unsorted_segment_max = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=tf.math.unsorted_segment_max)
batched_unsorted_segment_mean = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=tf.math.unsorted_segment_mean)
batched_unsorted_segment_sum = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=tf.math.unsorted_segment_sum)
batched_unsorted_segment_min = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=tf.math.unsorted_segment_min)


def points_to_voxels(points_xyz,
                     points_mask,
                     grid_size,
                     grid_range_x,
                     grid_range_y,
                     grid_range_z):
  """Mapping points to voxels."""
  batch_size, num_points = get_shape(points_xyz, 2)
  points_mask = tf.reshape(points_mask, [batch_size, num_points])
  voxel_size_x = (grid_range_x[1]-grid_range_x[0]) / grid_size[0]
  voxel_size_y = (grid_range_y[1]-grid_range_y[0]) / grid_size[1]
  voxel_size_z = (grid_range_z[1]-grid_range_z[0]) / grid_size[2]
  voxel_size = tf.convert_to_tensor([voxel_size_x, voxel_size_y, voxel_size_z],
                                    dtype=tf.dtypes.float32)

  num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
  grid_offset = tf.convert_to_tensor(
      [grid_range_x[0], grid_range_y[0], grid_range_z[0]],
      dtype=tf.dtypes.float32)
  points_xyz -= grid_offset
  voxel_xyz = points_xyz / voxel_size
  voxel_coords = tf.cast(voxel_xyz, tf.dtypes.int32)
  grid_size = tf.cast(grid_size, tf.dtypes.int32)
  zeros = tf.zeros_like(grid_size)
  voxel_padding = ((points_mask < 1.0) |
                   tf.reduce_any((voxel_coords >= grid_size) |
                                 (voxel_coords < zeros), axis=-1))
  voxel_indices = raval_index(
      tf.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
  voxel_indices = tf.reshape(voxel_indices, [batch_size, num_points])
  voxel_indices = tf.where(voxel_padding,
                           tf.zeros_like(voxel_indices),
                           voxel_indices)
  voxel_centers = (
      (0.5 + tf.cast(voxel_coords, tf.dtypes.float32)) * voxel_size +
      grid_offset)

  voxel_coords = tf.where(tf.expand_dims(voxel_padding, axis=-1),
                          tf.zeros_like(voxel_coords),
                          voxel_coords)
  voxel_xyz = tf.where(tf.expand_dims(voxel_padding, axis=-1),
                       tf.zeros_like(voxel_xyz),
                       voxel_xyz)

  voxel_padding = tf.cast(voxel_padding, tf.dtypes.float32)
  points_per_voxel = batched_unsorted_segment_sum(
      batched_data=tf.ones((batch_size, num_points), dtype=tf.int32),
      batched_segment_ids=voxel_indices,
      num_segments=num_voxels,
      batched_padding=voxel_padding)
  num_valid_voxels = tf.reduce_sum(tf.cast(
      tf.cast(points_per_voxel, tf.dtypes.bool), tf.dtypes.int32),
                                   axis=1)
  voxel_point_count = tf.gather(points_per_voxel,
                                voxel_indices,
                                batch_dims=1)

  output = {'coords': voxel_coords,
            'centers': voxel_centers,
            'indices': voxel_indices,
            'paddings': voxel_padding,
            'num_voxels': num_voxels,
            'grid_size': grid_size,
            'voxel_xyz': voxel_xyz,
            'voxel_point_count': voxel_point_count,
            'num_valid_voxels': num_valid_voxels,
            'points_per_voxel': points_per_voxel}
  return output


def points_to_voxels_stats(points_xyz, voxels):
  """Get additional features for points."""

  batch_size, num_points = get_shape(points_xyz, 2)

  # Compute centroids of each voxel.
  voxel_centroids = batched_unsorted_segment_mean(
      batched_data=points_xyz,
      batched_segment_ids=voxels['indices'],
      num_segments=voxels['num_voxels'],
      batched_padding=voxels['paddings'])
  point_centroids = tf.gather(voxel_centroids, voxels['indices'], batch_dims=1)
  points_xyz = points_xyz - point_centroids

  points_outer_prod = (
      points_xyz[..., :, tf.newaxis] * points_xyz[..., tf.newaxis, :])
  points_outer_prod = tf.reshape(points_outer_prod, [batch_size, num_points, 9])
  voxel_covariance = batched_unsorted_segment_mean(
      batched_data=points_outer_prod,
      batched_segment_ids=voxels['indices'],
      num_segments=voxels['num_voxels'],
      batched_padding=voxels['paddings'])
  points_covariance = tf.gather(voxel_covariance,
                                voxels['indices'],
                                batch_dims=1)

  output = {'centroids': point_centroids,
            'centered_xyz': points_xyz,
            'points_covariance': points_covariance,}

  return output


def points_xyz_to_cylinder(points_xyz):
  points_x, points_y, points_z = tf.unstack(points_xyz, axis=-1)
  points_rho = tf.math.sqrt(points_x**2 + points_y**2)
  points_phi = tf.math.atan2(points_y, points_x)
  points_cylinder = tf.stack([points_phi, points_z, points_rho], axis=-1)
  return points_cylinder


def points_cylinder_to_xyz(points_cylinder):
  points_phi, points_z, points_rho = tf.unstack(points_cylinder, axis=-1)
  points_x = points_rho * tf.math.cos(points_phi)
  points_y = points_rho * tf.math.sin(points_phi)
  points_xyz = tf.stack([points_x, points_y, points_z], axis=-1)
  return points_xyz


def wrap_angle_rad(angles_rad, min_val=-np.pi, max_val=np.pi):
  """Wrap the value of `angles_rad` to the range [min_val, max_val]."""
  max_min_diff = max_val - min_val
  return min_val + tf.math.floormod(angles_rad + max_val, max_min_diff)


def nms(bboxes, bbox_scores, nms_iou_threshold=0.7, nms_score_threshold=0.00,
        max_nms_boxes=200, use_oriented_per_class_nms=True):
  """NMS."""
  batch_size = get_shape(bboxes)[0]
  bboxes = tf.reshape(bboxes, [batch_size, -1, 7])
  bbox_scores = tf.reshape(bbox_scores, [batch_size, -1, 1])
  bbox_background = tf.zeros_like(bbox_scores)
  bbox_scores = tf.concat([bbox_background, bbox_scores], axis=-1)
  nms_bboxes, nms_bbox_scores, nms_valid_mask = (
      detection_decoder.DecodeWithNMS(
          bboxes,
          bbox_scores,
          nms_iou_threshold=nms_iou_threshold,
          score_threshold=nms_score_threshold,
          max_boxes_per_class=max_nms_boxes,
          use_oriented_per_class_nms=use_oriented_per_class_nms))
  nms_bboxes = tf.reshape(nms_bboxes[:, 1, :, :], [batch_size, -1, 7])
  nms_bbox_scores = tf.reshape(nms_bbox_scores[:, 1, :], [batch_size, -1])
  nms_valid_mask = tf.reshape(nms_valid_mask[:, 1, :], [batch_size, -1])
  nms_valid_mask = tf.cast(nms_valid_mask, tf.dtypes.int32)
  return nms_bboxes, nms_bbox_scores, nms_valid_mask


def _build_waymo_metric_config(metadata, box_type, waymo_breakdown_metrics):
  """Build the Config proto for Waymo's metric op."""
  config = metrics_pb2.Config()
  num_pr_points = metadata.NumberOfPrecisionRecallPoints()
  config.score_cutoffs.extend(
      [i * 1.0 / (num_pr_points - 1) for i in range(num_pr_points)])
  config.matcher_type = metrics_pb2.MatcherProto.Type.TYPE_HUNGARIAN
  if box_type == '2d':
    config.box_type = label_pb2.Label.Box.Type.TYPE_2D
  else:
    config.box_type = label_pb2.Label.Box.Type.TYPE_3D
  # Default values
  config.iou_thresholds[:] = [0.0, 0.7, 0.5, 0.5, 0.5]
  for class_name, threshold in metadata.IoUThresholds().items():
    cls_idx = metadata.ClassNames().index(class_name)
    config.iou_thresholds[cls_idx] = threshold
  config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.ONE_SHARD)
  config.difficulties.append(metrics_pb2.Difficulty())
  # Add extra breakdown metrics.
  for breakdown_value in waymo_breakdown_metrics:
    breakdown_id = breakdown_pb2.Breakdown.GeneratorId.Value(breakdown_value)
    config.breakdown_generator_ids.append(breakdown_id)
    config.difficulties.append(metrics_pb2.Difficulty())
  return config


def build_waymo_metric(pred_bbox, pred_class_id, pred_class_score,
                       pred_frame_id, gt_bbox, gt_class_id, gt_frame_id,
                       gt_speed, box_type='3d', breakdowns=None):
  """Build waymo evaluation metric."""
  metadata = waymo_metadata.WaymoMetadata()
  if breakdowns is None:
    breakdowns = ['RANGE']
  waymo_metric_config = _build_waymo_metric_config(
      metadata, box_type, breakdowns)

  ap, ap_ha, pr, pr_ha, _ = py_metrics_ops.detection_metrics(
      prediction_bbox=pred_bbox,
      prediction_type=pred_class_id,
      prediction_score=pred_class_score,
      prediction_frame_id=tf.cast(pred_frame_id, tf.int64),
      prediction_overlap_nlz=tf.zeros_like(pred_frame_id, dtype=tf.bool),
      ground_truth_bbox=gt_bbox,
      ground_truth_type=gt_class_id,
      ground_truth_frame_id=tf.cast(gt_frame_id, tf.int64),
      ground_truth_difficulty=tf.zeros_like(gt_frame_id, dtype=tf.uint8),
      ground_truth_speed=gt_speed,
      config=waymo_metric_config.SerializeToString())

  # All tensors returned by Waymo's metric op have a leading dimension
  # B=number of breakdowns. At this moment we always use B=1 to make
  # it compatible to the python code.
  scalar_metrics = {'%s_ap' % box_type: ap[0],
                    '%s_ap_ha_weighted' % box_type: ap_ha[0]}
  curve_metrics = {'%s_pr' % box_type: pr[0],
                   '%s_pr_ha_weighted'% box_type: pr_ha[0]}

  breakdown_names = config_util.get_breakdown_names_from_config(
      waymo_metric_config)
  for i, metric in enumerate(breakdown_names):
    # There is a scalar / curve for every breakdown.
    scalar_metrics['%s_ap_%s' % (box_type, metric)] = ap[i]
    scalar_metrics['%s_ap_ha_weighted_%s' % (box_type, metric)] = ap_ha[i]
    curve_metrics['%s_pr_%s' % (box_type, metric)] = pr[i]
    curve_metrics['%s_pr_ha_weighted_%s' % (box_type, metric)] = pr_ha[i]
  return scalar_metrics, curve_metrics


def compute_ap(decoded_outputs, class_id):
  """Compute average precision."""
  pred_bbox = decoded_outputs['bboxes_pred']
  pred_class_score = decoded_outputs['bboxes_pred_score']
  pred_bbox_mask = decoded_outputs['bboxes_pred_mask']
  batch_size, num_bboxes = get_shape(pred_bbox, 2)
  pred_frame_id = tf.range(batch_size)
  pred_frame_id = tf.reshape(pred_frame_id, [batch_size, 1])
  pred_frame_id = tf.tile(pred_frame_id, [1, num_bboxes])

  gt_bbox = decoded_outputs['bboxes']
  gt_bbox_mask = decoded_outputs['bboxes_mask']
  gt_bbox_speed = decoded_outputs['bboxes_speed']

  batch_size, num_bboxes = get_shape(gt_bbox, 2)
  gt_frame_id = tf.range(batch_size)
  gt_frame_id = tf.reshape(gt_frame_id, [batch_size, 1])
  gt_frame_id = tf.tile(gt_frame_id, [1, num_bboxes])

  pred_bbox = tf.reshape(pred_bbox, [-1, 7])
  pred_class_score = tf.reshape(pred_class_score, [-1])
  pred_frame_id = tf.reshape(pred_frame_id, [-1])
  pred_bbox_mask = tf.reshape(pred_bbox_mask, [-1])

  pred_bbox = pred_bbox[pred_bbox_mask == 1]
  pred_class_score = pred_class_score[pred_bbox_mask == 1]
  pred_frame_id = pred_frame_id[pred_bbox_mask == 1]
  num_pd_bboxes = get_shape(pred_bbox)[0]
  pred_class_id = tf.constant(class_id, dtype=tf.dtypes.uint8,
                              shape=[num_pd_bboxes])

  gt_bbox = tf.reshape(gt_bbox, [-1, 7])
  gt_bbox_speed = tf.reshape(gt_bbox_speed, [-1, 2])
  gt_frame_id = tf.reshape(gt_frame_id, [-1])
  gt_bbox_mask = tf.reshape(gt_bbox_mask, [-1])

  gt_bbox = gt_bbox[gt_bbox_mask == 1]
  gt_bbox_speed = gt_bbox_speed[gt_bbox_mask == 1]
  gt_frame_id = gt_frame_id[gt_bbox_mask == 1]
  num_gt_bboxes = get_shape(gt_bbox)[0]

  gt_class_id = tf.constant(class_id, dtype=tf.dtypes.uint8,
                            shape=[num_gt_bboxes])

  scalar_metrics_3d, _ = build_waymo_metric(
      pred_bbox, pred_class_id, pred_class_score, pred_frame_id,
      gt_bbox, gt_class_id, gt_frame_id, gt_bbox_speed)
  scalar_metrics_2d, _ = build_waymo_metric(
      pred_bbox, pred_class_id, pred_class_score, pred_frame_id,
      gt_bbox, gt_class_id, gt_frame_id, gt_bbox_speed, box_type='2d')
  return scalar_metrics_3d, scalar_metrics_2d
