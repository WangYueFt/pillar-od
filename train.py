"""Train."""
import os

from absl import app
from absl import logging
import numpy as np
import config
import network as builder
import waymo_loader
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

FLAGS = config.define_flags()

STEPS_PER_LOOP = 30
_SUMMARY_TXT = 'training_summary.txt'
_MIN_SUMMARY_STEPS = 10


def get_strategy():
  if FLAGS.use_tpu:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.master)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.MirroredStrategy()
  return strategy


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
  """Saves model to with provided checkpoint prefix."""

  checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
  saved_path = checkpoint.save(checkpoint_path)
  logging.info('Saving model as TF checkpoint: %s', saved_path)
  return


def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  if steps_per_loop == 1:
    return steps_per_loop
  remainder_in_epoch = current_step % steps_per_epoch
  if remainder_in_epoch != 0:
    return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
  else:
    return steps_per_loop


def _float_metric_value(metric):
  """Gets the value of a float-value keras metric."""
  return metric.result().numpy().astype(float)


class CosineLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self,
               initial_learning_rate,
               min_learning_rate,
               max_learning_rate,
               warmup_steps,
               steps_per_epoch,
               max_steps):
    super(CosineLearningRateSchedule, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.min_learning_rate = min_learning_rate
    self.max_learning_rate = max_learning_rate
    self.steps_per_epoch = steps_per_epoch
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps

  def __call__(self, step):
    step = tf.cast(step, tf.dtypes.float32)
    warmup_learning_rate = (
        self.max_learning_rate -
        self.initial_learning_rate) / self.warmup_steps * step
    warmup_learning_rate = self.initial_learning_rate + warmup_learning_rate
    cosine_learning_rate = (
        self.min_learning_rate +
        (self.max_learning_rate - self.min_learning_rate) *
        (1 + tf.math.cos(step * np.pi / self.max_steps)) / 2)
    learning_rate = tf.where(step > self.warmup_steps, cosine_learning_rate,
                             warmup_learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'min_learning_rate': self.min_learning_rate,
        'max_learning_rate': self.max_learning_rate,
        'steps_per_epoch': self.steps_per_epoch,
        'warmup_steps': self.warmup_steps,
        'max_steps': self.max_steps,
    }


def main(_):
  strategy = get_strategy()
  num_replicas_in_sync = strategy.num_replicas_in_sync
  global_batch_size = (FLAGS.batch_size *
                       num_replicas_in_sync)
  num_train_samples = 158361
  steps_per_epoch = num_train_samples // global_batch_size
  total_training_steps = FLAGS.epochs  * steps_per_epoch

  with strategy.scope():
    # Make a model
    model = builder.PillarModel(
        class_id=FLAGS.class_id,
        norm_type=FLAGS.norm_type,
        act_type=FLAGS.act_type,
        nms_iou_threshold=FLAGS.nms_iou_threshold,
        nms_score_threshold=FLAGS.nms_score_threshold,
        max_nms_boxes=FLAGS.max_nms_boxes,
        use_oriented_per_class_nms=FLAGS.use_oriented_per_class_nms)
    ema_model = builder.PillarModel(
        class_id=FLAGS.class_id,
        norm_type=FLAGS.norm_type,
        act_type=FLAGS.act_type,
        nms_iou_threshold=FLAGS.nms_iou_threshold,
        nms_score_threshold=FLAGS.nms_score_threshold,
        max_nms_boxes=FLAGS.max_nms_boxes,
        use_oriented_per_class_nms=FLAGS.use_oriented_per_class_nms)
    lr = FLAGS.lr

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CosineLearningRateSchedule(
            lr * 0.1,
            lr * 0.001,
            lr,
            steps_per_epoch * 1,
            steps_per_epoch,
            total_training_steps))

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loc_loss_metric = tf.keras.metrics.Mean('train_loc_loss',
                                                  dtype=tf.float32)
    train_cls_loss_metric = tf.keras.metrics.Mean('train_cls_loss',
                                                  dtype=tf.float32)
    weight_loss_metric = tf.keras.metrics.Mean('weight_loss',
                                               dtype=tf.float32)
    # Create summary writers
    model_dir = FLAGS.model_dir
    summary_dir = os.path.join(model_dir, 'summaries')
    steps_per_loop = STEPS_PER_LOOP
    if steps_per_loop >= _MIN_SUMMARY_STEPS:
      # Only writes summary when the stats are collected sufficiently over
      # enough steps.
      train_summary_writer = tf.summary.create_file_writer(
          os.path.join(summary_dir, 'train'))
    else:
      train_summary_writer = None

    # Make a dataset
    dataset_train = waymo_loader.waymo_open_dataset(
        data_path=FLAGS.data_path,
        batch_size=global_batch_size,
        cycle_length=FLAGS.cycle_length,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        num_parallel_calls=FLAGS.num_parallel_calls,
        percentile=1.0,
        max_num_points=FLAGS.max_num_points,
        max_num_bboxes=FLAGS.max_num_bboxes,
        class_id=FLAGS.class_id,
        difficulty=FLAGS.difficulty,
        pillar_map_size=(FLAGS.pillar_map_size, FLAGS.pillar_map_size),
        pillar_map_range=(FLAGS.pillar_map_range, FLAGS.pillar_map_range))

    dist_dataset_train = strategy.experimental_distribute_dataset(dataset_train)
    train_iterator = iter(dist_dataset_train)

    def _momentum_update(model, ema_model, momentum=0.999,
                         just_trainable_vars=False):
      """Update the momentum encoder."""

      replica_context = tf.distribute.get_replica_context()
      iterable = (
          zip(model.trainable_variables, ema_model.trainable_variables)
          if just_trainable_vars
          else zip(model.variables, ema_model.variables)
          )
      values_and_vars = []
      for p, p2 in iterable:
        v = momentum * p2 + (1.0 - momentum) * p
        values_and_vars.append((v, p2))

      def _distributed_update(strategy, values_and_vars):
        reduced_values = strategy.extended.batch_reduce_to(
            tf.distribute.ReduceOp.MEAN, values_and_vars)
        variables = [v for _, v in values_and_vars]
        def _update(var, value):
          var.assign(value)
        for var, value in zip(variables, reduced_values):
          strategy.extended.update(
              var, _update, args=(value,))
      replica_context.merge_call(_distributed_update, args=(values_and_vars,))

    def _replicated_step(inputs):
      """Replicated training step."""

      with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        if optimizer.iterations < 1:
          ema_model(inputs, training=False)
        cls_loss, loc_loss = model.compute_loss(inputs, preds)

        mean_cls_loss = tf.reduce_mean(cls_loss)
        mean_loc_loss = tf.reduce_mean(loc_loss)
        weight_loss = tf.reduce_sum(model.losses)
        loss = (mean_cls_loss + mean_loc_loss + weight_loss)
        training_vars = model.trainable_variables
        grads = tape.gradient(loss, training_vars)

      optimizer.apply_gradients(zip(grads, training_vars))
      train_loss_metric.update_state(loss)
      train_loc_loss_metric.update_state(mean_loc_loss)
      train_cls_loss_metric.update_state(mean_cls_loss)
      weight_loss_metric.update_state(weight_loss)
      _momentum_update(model, ema_model)
      return loss

    @tf.function
    def train_steps(iterator, steps):
      """Performs distributed training steps in a loop.

      Args:
        iterator: the distributed iterator of training datasets.
        steps: an tf.int32 integer tensor to specify number of steps to run
          inside host training loop.

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
      if not isinstance(steps, tf.Tensor):
        raise ValueError('steps should be an Tensor. Python object may cause '
                         'retracing.')

      for _ in tf.range(steps):
        per_replica_loss = strategy.experimental_run_v2(
            _replicated_step, args=(next(iterator),))
        strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    def train_single_step(iterator):
      """Performs a distributed training step.

      Args:
        iterator: the distributed iterator of training datasets.

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
      per_replica_loss = strategy.experimental_run_v2(
          _replicated_step, args=(next(iterator),))
      strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,
                      axis=None)

    latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
    # Training loop starts here.
    checkpoint = tf.train.Checkpoint(model=model, ema_model=ema_model,
                                     optimizer=optimizer)
    if latest_checkpoint_file:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'checkpoint', latest_checkpoint_file)
      checkpoint.restore(latest_checkpoint_file)
      logging.info('Loading from checkpoint file completed')

    current_step = optimizer.iterations.numpy()
    checkpoint_name = 'ctl_step_{step}.ckpt'

    while current_step < total_training_steps:
      # Training loss/metric are taking average over steps inside micro
      # training loop. We reset the their values before each round.
      steps = steps_to_run(current_step, steps_per_epoch, steps_per_loop)

      if steps == 1:
        train_single_step(train_iterator)
      else:
        # Converts steps to a Tensor to avoid tf.function retracing.
        train_steps(train_iterator,
                    tf.convert_to_tensor(steps, dtype=tf.int32))
      current_step += steps
      train_loss = _float_metric_value(train_loss_metric)
      train_cls_loss = _float_metric_value(train_cls_loss_metric)
      train_loc_loss = _float_metric_value(train_loc_loss_metric)

      weight_loss = _float_metric_value(weight_loss_metric)

      # Updates training logging.
      lr = optimizer.lr(optimizer.iterations).numpy()
      # lr = optimizer.lr.lr()
      training_status = (
          'Train Step: %d/%d, loss = %s, cls_loss = %s, loc_loss = %s, '
          'weight_loss = %s, lr = %f' %
          (current_step, total_training_steps, train_loss, train_cls_loss,
           train_loc_loss, weight_loss, lr))

      if train_summary_writer:
        with train_summary_writer.as_default():
          tf.summary.scalar(
              train_loss_metric.name, train_loss, step=current_step)
          tf.summary.scalar(
              train_cls_loss_metric.name, train_cls_loss, step=current_step)
          tf.summary.scalar(
              train_loc_loss_metric.name, train_loc_loss, step=current_step)
          tf.summary.scalar(
              weight_loss_metric.name, weight_loss, step=current_step)
          tf.summary.scalar(
              'lr', lr, step=current_step)
          train_summary_writer.flush()
      logging.info(training_status)

      # Saves model checkpoints and run validation steps at every epoch end.
      if current_step % steps_per_epoch == 0:
        # To avoid repeated model saving, we do not save after the last
        # step of training.
        # if current_step < total_training_steps:
        _save_checkpoint(checkpoint, model_dir,
                         checkpoint_name.format(step=current_step))


if __name__ == '__main__':
  app.run(main)
