"""Network module."""

from absl import logging
import numpy as np
import loss
import tf_util
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa
tf.enable_v2_behavior()


KERNEL_REGULARIZER = tf.keras.regularizers.l2(0.0)


def dense(odims=64, use_bias=False):
  return tf.keras.layers.Dense(odims,
                               kernel_initializer='he_uniform',
                               kernel_regularizer=KERNEL_REGULARIZER,
                               use_bias=use_bias)


def conv1d(odims=64, kernel_size=1, stride=1, use_bias=False):
  return tf.keras.layers.Conv1D(odims,
                                kernel_size,
                                strides=stride,
                                padding='same',
                                kernel_initializer='he_uniform',
                                kernel_regularizer=KERNEL_REGULARIZER,
                                use_bias=use_bias)


def conv2d(odims=64, kernel_size=(3, 3), stride=1, use_bias=True):
  return tf.keras.layers.Conv2D(odims,
                                kernel_size,
                                strides=stride,
                                padding='same',
                                kernel_initializer='he_uniform',
                                kernel_regularizer=KERNEL_REGULARIZER,
                                use_bias=use_bias)


def deconv2d(odims=64, kernel_size=(3, 3), stride=1):
  return tf.keras.layers.Conv2DTranspose(
      odims,
      kernel_size,
      strides=stride,
      padding='same',
      kernel_initializer='he_uniform',
      kernel_regularizer=KERNEL_REGULARIZER)


def get_normalization(norm_type='sync_batch_norm'):
  """Get normalization."""
  if norm_type == 'batch_norm':
    norm_layer = tf.keras.layers.BatchNormalization()
  elif norm_type == 'sync_batch_norm':
    norm_layer = SyncBatchNormalization()
  elif norm_type == 'layer_norm':
    norm_layer = tf.keras.layers.LayerNormalization()
  elif norm_type == 'none':
    norm_layer = identity()
  else:
    raise NotImplementedError(norm_layer)
  return norm_layer


def get_activation(act_type='relu'):
  if act_type == 'relu':
    act_layer = tf.keras.layers.ReLU(negative_slope=0)
  elif act_type == 'leakyrelu':
    act_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
  elif act_type == 'elu':
    act_layer = tf.keras.layers.ELU(alpha=1.0)
  else:
    raise NotImplementedError(act_layer)
  return act_layer


def identity():
  return tf.keras.layers.Lambda(lambda x: x)


class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
  """Batch Normalization layer that supports cross replica computation.

  This class extends the keras.BatchNormalization implementation by supporting
  cross replica means and variances. The base class implementation only computes
  moments based on mini-batch per replica.

  For detailed information of arguments and implementation, refer to:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  """

  def __init__(self, fused=False, **kwargs):
    """Builds the batch normalization layer.

    Arguments:
      fused: If `False`, use the system recommended implementation. Only support
        `False` in the current implementation.
      **kwargs: input augments that are forwarded to
        tf.layers.BatchNormalization.
    """
    if fused in (True, None):
      raise ValueError('This version of BatchNormalization does not support '
                       'fused=True.')
    super(SyncBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t):
    """Calculates the average value of input tensor across replicas."""
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None:
      raise TypeError(
          'Cross replica batch norm cannot be called in cross-replica context.')
    return replica_context.all_reduce('MEAN', t)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(SyncBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tf.distribute.get_replica_context().num_replicas_in_sync
    logging.info('BatchNormalization with num_shards_per_group %s',
                 num_shards)

    group_mean = self._cross_replica_average(shard_mean)
    group_variance = self._cross_replica_average(shard_variance)

    # Group variance needs to also include the difference between shard_mean
    # and group_mean.
    mean_distance = tf.square(group_mean - shard_mean)
    group_variance += self._cross_replica_average(mean_distance)
    return (group_mean, group_variance)


class BasicBlock(tf.keras.layers.Layer):
  """ResNet Basic Block."""

  def __init__(self, idims, odims, kernel_size=(3, 3), stride=1,
               norm_type='sync_batch_norm', act_type='relu'):
    super(BasicBlock, self).__init__()
    self.conv1 = conv2d(odims, kernel_size, stride)
    self.bn1 = get_normalization(norm_type)
    self.relu1 = get_activation(act_type)
    self.conv2 = conv2d(odims, kernel_size, 1)
    self.bn2 = get_normalization(norm_type)
    self.relu2 = get_activation(act_type)
    if idims == odims and stride == 1:
      self.shortcut = identity()
    else:
      self.shortcut = tf.keras.Sequential(
          [conv2d(odims, (1, 1), stride),
           get_normalization()])

  def call(self, x):
    shortcut = self.shortcut(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + shortcut
    x = self.relu2(x)
    return x


class PillarBlock(tf.keras.layers.Layer):
  """Pillar Block."""

  def __init__(self, base_dims=64, dim_factor=2, num_layers=1,
               stride=1, norm_type='sync_batch_norm', act_type='relu',
               use_res_block=False):
    super(PillarBlock, self).__init__()
    dims = base_dims * dim_factor
    blocks = []
    for i in range(num_layers):
      if i != 0:
        stride = 1
      if use_res_block:
        blocks.append(BasicBlock(dims, dims, stride=stride,
                                 norm_type=norm_type, act_type=act_type))
      else:
        blocks.append(conv2d(stride=stride, odims=dims))
        blocks.append(get_normalization(norm_type))
        blocks.append(get_activation(act_type))
    self.blocks = tf.keras.Sequential(blocks)

  def call(self, x):
    return self.blocks(x)


class PointNet(tf.keras.layers.Layer):
  """PointNet."""

  def __init__(self, odims=64, norm_type='sync_batch_norm', act_type='relu'):
    super(PointNet, self).__init__()
    if isinstance(odims, list):
      layers = []
      for odim in odims:
        layers.append(dense(odims=odim))
        layers.append(get_normalization(norm_type))
        layers.append(get_activation(act_type))
      self.pointnet = tf.keras.Sequential(layers)
    else:
      self.pointnet = tf.keras.Sequential([
          dense(odims=odims),
          get_normalization(norm_type),
          get_activation(act_type)
      ])

  def call(self, points_feature, points_mask):
    batch_size, num_points = tf_util.get_shape(points_feature, 2)
    points_mask = tf.reshape(points_mask, [batch_size, num_points, 1])
    points_feature = self.pointnet(points_feature) * points_mask
    return points_feature


class SingleViewNet(tf.keras.layers.Layer):
  """SingleViewNet.

     Bird view or Cylinderical view.
  """

  def __init__(self, grid_size=(512, 512, 1),
               norm_type='sync_batch_norm', act_type='relu'):

    super(SingleViewNet, self).__init__()
    self.pointnet = PointNet(odims=128, norm_type=norm_type, act_type=act_type)
    self.res1 = BasicBlock(idims=128, odims=128,
                           norm_type=norm_type, act_type=act_type)
    self.res2 = BasicBlock(idims=128, odims=128, stride=2,
                           norm_type=norm_type, act_type=act_type)
    self.res3 = BasicBlock(idims=128, odims=128, stride=2,
                           norm_type=norm_type, act_type=act_type)
    self.deconv2 = deconv2d(odims=128, stride=2)
    self.deconv3 = deconv2d(odims=128, stride=4)
    self.conv = conv2d(odims=128, stride=1)
    self.grid_size = [x for x in grid_size if x > 1]

  def call(self, points_xyz, points_feature, points_mask, points_voxel):
    batch_size, _ = tf_util.get_shape(points_feature, 2)
    points_feature = self.pointnet(points_feature, points_mask)
    voxels = tf_util.batched_unsorted_segment_max(
        batched_data=points_feature,
        batched_segment_ids=points_voxel['indices'],
        num_segments=points_voxel['num_voxels'],
        batched_padding=points_voxel['paddings'])

    _, _, nc = tf_util.get_shape(voxels)

    voxels = tf.reshape(voxels, [-1])
    voxels = tf.where(voxels <= voxels.dtype.min, tf.zeros_like(voxels), voxels)
    voxels_in = tf.reshape(voxels, [batch_size] + self.grid_size + [nc])
    voxels_out1 = self.res1(voxels_in)
    voxels_out2 = self.res2(voxels_in)
    voxels_out3 = self.res3(voxels_out2)
    voxels_out2 = self.deconv2(voxels_out2)
    voxels_out3 = self.deconv3(voxels_out3)
    voxels_out = tf.concat([voxels_out1, voxels_out2, voxels_out3], axis=-1)
    voxels_out = self.conv(voxels_out)
    nc = tf_util.get_shape(voxels_out)[-1]

    # w/ bilinear interpolation
    points_out = tfa.image.interpolate_bilinear(
        voxels_out, points_voxel['voxel_xyz'][:, :, :2])

    return points_out


class PillarNet(tf.keras.layers.Layer):
  """Pillar Net."""

  def __init__(self, norm_type='sync_batch_norm', act_type='relu',
               stride=(2, 1, 2), up_stride=(1, 1, 2)):
    super(PillarNet, self).__init__()

    self.xy_view_grid_size = (512, 512, 1)
    self.xy_view_grid_range_x = (-75.2, 75.2)
    self.xy_view_grid_range_y = (-75.2, 75.2)
    self.xy_view_grid_range_z = (-5.0, 5.0)

    self.cylinder_view_grid_size = (2560, 100, 1)
    self.cylinder_view_grid_range_x = (-np.pi, np.pi)
    self.cylinder_view_grid_range_y = (-3.0, 7.0)
    self.cylinder_view_grid_range_z = (0.0, 107.0)

    self.xy_view = SingleViewNet(self.xy_view_grid_size,
                                 norm_type=norm_type,
                                 act_type=act_type)
    self.cylinder_view = SingleViewNet(self.cylinder_view_grid_size,
                                       norm_type=norm_type,
                                       act_type=act_type)

    self.pointnet1 = PointNet(odims=128,
                              norm_type=norm_type,
                              act_type=act_type)
    self.pointnet2 = PointNet(odims=128,
                              norm_type=norm_type,
                              act_type=act_type)
    self.pointnet3 = PointNet(odims=128,
                              norm_type=norm_type,
                              act_type=act_type)
    self.block1 = PillarBlock(base_dims=64, dim_factor=2,
                              num_layers=4, stride=stride[0],
                              norm_type=norm_type, act_type=act_type)

    self.up1 = tf.keras.Sequential([deconv2d(128, stride=up_stride[0]),
                                    get_normalization(norm_type),
                                    get_activation(act_type)])

    self.block2 = PillarBlock(base_dims=64, dim_factor=2,
                              num_layers=6, stride=stride[1],
                              norm_type=norm_type, act_type=act_type)

    self.up2 = tf.keras.Sequential([deconv2d(128, stride=up_stride[1]),
                                    get_normalization(norm_type),
                                    get_activation(act_type)])
    self.block3 = PillarBlock(base_dims=64, dim_factor=4,
                              num_layers=6, stride=stride[2],
                              norm_type=norm_type, act_type=act_type)

    self.up3 = tf.keras.Sequential([deconv2d(256, stride=up_stride[2]),
                                    get_normalization(norm_type),
                                    get_activation(act_type)])
    self.conv = tf.keras.Sequential([conv2d(256, stride=1),
                                     get_normalization(norm_type),
                                     get_activation(act_type)])

  def call(self, inputs):
    points_xyz = inputs['points_xyz']
    points_feature = inputs['points_feature']
    points_mask = inputs['points_mask']
    batch_size, num_points = tf_util.get_shape(points_xyz, 2)
    xy_view_voxels = tf_util.points_to_voxels(points_xyz,
                                              points_mask,
                                              self.xy_view_grid_size,
                                              self.xy_view_grid_range_x,
                                              self.xy_view_grid_range_y,
                                              self.xy_view_grid_range_z)
    xy_view_voxels_stats = tf_util.points_to_voxels_stats(points_xyz,
                                                          xy_view_voxels)
    xy_view_points_xyz = points_xyz - xy_view_voxels['centers']

    points_cylinder = tf_util.points_xyz_to_cylinder(points_xyz)
    cylinder_view_voxels = tf_util.points_to_voxels(
        points_cylinder, points_mask, self.cylinder_view_grid_size,
        self.cylinder_view_grid_range_x,
        self.cylinder_view_grid_range_y,
        self.cylinder_view_grid_range_z)
    cylinder_view_voxels_stats = tf_util.points_to_voxels_stats(
        points_cylinder, cylinder_view_voxels)
    cylinder_view_points = points_cylinder - cylinder_view_voxels['centers']

    points_feature = tf.concat([
        points_xyz,
        xy_view_points_xyz,
        tf.cast(tf.reshape(xy_view_voxels['voxel_point_count'],
                           [batch_size, num_points, 1]),
                tf.dtypes.float32),
        xy_view_voxels_stats['centered_xyz'],
        xy_view_voxels_stats['points_covariance'],
        xy_view_voxels_stats['centroids'],
        points_cylinder,
        cylinder_view_points,
        tf.cast(tf.reshape(cylinder_view_voxels['voxel_point_count'],
                           [batch_size, num_points, 1]),
                tf.dtypes.float32),
        cylinder_view_voxels_stats['centered_xyz'],
        cylinder_view_voxels_stats['points_covariance'],
        cylinder_view_voxels_stats['centroids'],
        points_feature], axis=-1)
    x = self.pointnet1(points_feature, points_mask)

    x_xy_view = self.xy_view(points_xyz,
                             x,
                             points_mask,
                             xy_view_voxels)

    x_cylinder_view = self.cylinder_view(points_cylinder,
                                         x,
                                         points_mask,
                                         cylinder_view_voxels)

    x_pointwise = self.pointnet2(x, points_mask)
    x = tf.concat([
        x_xy_view,
        x_cylinder_view,
        x_pointwise], axis=-1)
    x = self.pointnet3(x, points_mask)

    pillars = tf_util.batched_unsorted_segment_max(
        batched_data=x,
        batched_segment_ids=xy_view_voxels['indices'],
        num_segments=xy_view_voxels['num_voxels'],
        batched_padding=xy_view_voxels['paddings'])

    _, _, nc = tf_util.get_shape(pillars)
    pillars = tf.reshape(pillars, [-1])
    pillars = tf.where(pillars <= pillars.dtype.min,
                       tf.zeros_like(pillars),
                       pillars)
    nx, ny, nz = self.xy_view_grid_size
    pillars = tf.reshape(pillars, [batch_size, nx, ny, nz * nc])
    out1 = self.block1(pillars)
    out2 = self.block2(out1)
    out3 = self.block3(out2)
    out1 = self.up1(out1)
    out2 = self.up2(out2)
    out3 = self.up3(out3)
    out = tf.concat([out1, out2, out3], axis=-1)
    out = self.conv(out)
    return out


class PillarModel(tf.keras.Model):
  """Pillar Model."""

  def __init__(self, class_id=1, norm_type='sync_batch_norm', act_type='relu',
               nms_iou_threshold=0.7, nms_score_threshold=0.00,
               max_nms_boxes=200, use_oriented_per_class_nms=True):
    super(PillarModel, self).__init__()

    self.nms_iou_threshold = nms_iou_threshold
    self.nms_score_threshold = nms_score_threshold
    self.max_nms_boxes = max_nms_boxes
    self.use_oriented_per_class_nms = use_oriented_per_class_nms

    if class_id == 1:
      self.size_prior = [4.5, 2.0, 1.6]
      self.stride = (2, 1, 2)
      self.up_stride = (1, 1, 2)

    elif class_id == 2:
      self.size_prior = [0.6, 0.8, 1.8]
      self.stride = (1, 2, 2)
      self.up_stride = (1, 2, 4)

    else:
      raise NotImplementedError(class_id)

    self.pillar_net = PillarNet(norm_type=norm_type, act_type=act_type,
                                stride=self.stride, up_stride=self.up_stride)

    self.cls_mlp = tf.keras.Sequential([
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256)])
    self.loc_mlp = tf.keras.Sequential([
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256, use_bias=False),
        get_normalization(norm_type),
        get_activation(act_type),
        conv2d(256)])
    self.cls_fc = dense(odims=1)
    self.loc_fc = dense(odims=7)

  def call(self, inputs):
    features = self.pillar_net(inputs)

    cls_features = self.cls_mlp(features)
    loc_features = self.loc_mlp(features)

    cls_preds = self.cls_fc(cls_features)
    reg_preds = self.loc_fc(loc_features)

    batch_size, nx, ny, _ = tf_util.get_shape(reg_preds)
    cls_logits = tf.reshape(cls_preds, [batch_size, nx*ny, 1])
    reg_logits = tf.reshape(reg_preds, [batch_size, nx*ny, -1])

    preds = {
        'cls_logits': cls_logits,
        'reg_logits': reg_logits,
    }

    return preds

  def infer(self, inputs, preds):
    cls_logits = preds['cls_logits']
    reg_logits = preds['reg_logits']
    reg_xyz = inputs['pillar_map_xyz']
    cls_preds = tf.math.sigmoid(cls_logits)
    size_prior = tf.convert_to_tensor(self.size_prior)
    size_prior = tf.reshape(size_prior, [1, 1, 3])

    pos_logits = reg_logits[:, :, 0:3]
    size_logits = reg_logits[:, :, 3:6]
    angle_logits = reg_logits[:, :, 6:7]

    size_preds = tf.math.exp(size_logits) * size_prior
    pos_preds = pos_logits * size_prior + reg_xyz
    angle_preds = tf_util.wrap_angle_rad(angle_logits, 0, np.pi)
    loc_preds = tf.concat([pos_preds, size_preds, angle_preds], axis=-1)

    loc_preds, cls_preds, loc_mask = tf_util.nms(
        loc_preds, cls_preds,
        nms_iou_threshold=self.nms_iou_threshold,
        nms_score_threshold=self.nms_score_threshold,
        max_nms_boxes=self.max_nms_boxes,
        use_oriented_per_class_nms=self.use_oriented_per_class_nms)
    return {'loc_preds': loc_preds,
            'cls_preds': cls_preds,
            'loc_mask': loc_mask}

  def compute_loss(self, inputs, preds):
    reg_labels = inputs['pillar_map_bboxes']
    cls_labels = inputs['pillar_map_if_in_bboxes']
    reg_xyz = inputs['pillar_map_xyz']
    reg_logits = preds['reg_logits']
    cls_logits = preds['cls_logits']
    cls_loss = loss.sigmoid_cross_entropy_focal_loss(cls_logits, cls_labels)
    cls_loss = tf.reduce_sum(cls_loss, axis=1)

    angle_pred = reg_logits[:, :, 6]
    angle_label = reg_labels[:, :, 6]

    rot_diff = tf.sin(angle_label - angle_pred)
    rot_loss = tf.math.multiply_no_nan(
        loss.smooth_l1_loss(rot_diff, tf.zeros_like(rot_diff)), cls_labels)

    size_prior = tf.convert_to_tensor(self.size_prior)
    size_prior = tf.reshape(size_prior, [1, 1, 3])
    size_pred = reg_logits[:, :, 3:6]
    size_label = reg_labels[:, :, 3:6]

    size_loss = tf.reduce_sum(
        loss.smooth_l1_loss(size_pred, tf.math.log(
            tf.clip_by_value(size_label / size_prior, 1e-8, 1e10))),
        axis=-1)
    size_loss = tf.math.multiply_no_nan(size_loss, cls_labels)

    pos_pred = reg_logits[:, :, 0:3]
    pos_label = reg_labels[:, :, 0:3]
    pos_loss = tf.reduce_sum(
        loss.smooth_l1_loss(
            pos_pred,
            (pos_label - reg_xyz) / size_prior),
        axis=-1)
    pos_loss = tf.math.multiply_no_nan(pos_loss, cls_labels)
    reg_loss = rot_loss + size_loss + pos_loss

    has_pos = tf.reduce_max(cls_labels, axis=1)
    reg_loss = tf.reduce_sum(reg_loss, axis=1)
    reg_loss = tf.math.multiply_no_nan(reg_loss, has_pos)

    return cls_loss, reg_loss
