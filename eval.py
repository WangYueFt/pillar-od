"""Eval."""
import os
import time

from absl import app
from absl import logging
import config
import network as builder
import tf_util
import waymo_loader
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

FLAGS = config.define_flags()

_SUMMARY_TXT = 'validation_summary.txt'
_MIN_SUMMARY_STEPS = 10


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


def main(_):
  batch_size = FLAGS.test_batch_size

  # Fake optimizer
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr, clipnorm=10.0)

  # Make a model
  model = builder.PillarModel(
      class_id=FLAGS.class_id,
      norm_type=FLAGS.norm_type,
      act_type=FLAGS.act_type,
      nms_iou_threshold=FLAGS.nms_iou_threshold,
      nms_score_threshold=FLAGS.nms_score_threshold,
      max_nms_boxes=FLAGS.max_nms_boxes,
      use_oriented_per_class_nms=FLAGS.use_oriented_per_class_nms)

  # Create summary writers
  model_dir = FLAGS.model_dir
  summary_dir = os.path.join(model_dir, 'summaries')
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(summary_dir, 'eval'))

  # Make a dataset
  dataset_val = waymo_loader.waymo_open_dataset(
      data_path=FLAGS.data_path,
      batch_size=batch_size,
      split='valid',
      cycle_length=FLAGS.cycle_length,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      num_parallel_calls=FLAGS.num_parallel_calls,
      percentile=FLAGS.percentile,
      max_num_points=FLAGS.max_num_points,
      max_num_bboxes=FLAGS.max_num_bboxes,
      class_id=FLAGS.class_id,
      difficulty=FLAGS.difficulty,
      pillar_map_size=(FLAGS.pillar_map_size, FLAGS.pillar_map_size),
      pillar_map_range=(FLAGS.pillar_map_range, FLAGS.pillar_map_range))

  checkpoint_file = None
  while True:
    # Validation loop starts here.
    checkpoint = tf.train.Checkpoint(ema_model=model, optimizer=optimizer)
    if FLAGS.ckpt_path and FLAGS.eval_once:
      latest_checkpoint_file = FLAGS.ckpt_path
    else:
      latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)

    if latest_checkpoint_file == checkpoint_file:
      time.sleep(60)
      continue
    else:
      logging.info(
          'Checkpoint file %s found and restoring from '
          'checkpoint', latest_checkpoint_file)
      checkpoint.restore(latest_checkpoint_file)
      logging.info('Loading from checkpoint file completed')
      checkpoint_file = latest_checkpoint_file

    current_step = optimizer.iterations.numpy()

    total_loss = 0
    total_cls_loss = 0
    total_loc_loss = 0
    total_example = 0

    bboxes_pred = []
    bboxes_pred_score = []
    bboxes_pred_mask = []

    bboxes = []
    bboxes_mask = []
    bboxes_speed = []

    for inputs in dataset_val:
      preds = model(inputs, training=False)
      outputs = model.infer(inputs, preds)
      bboxes_pred.append(outputs['loc_preds'])
      bboxes_pred_score.append(outputs['cls_preds'])
      bboxes_pred_mask.append(outputs['loc_mask'])

      bboxes.append(inputs['bboxes'])
      bboxes_mask.append(inputs['bboxes_mask'])
      bboxes_speed.append(inputs['bboxes_speed'])

      batch_size = tf_util.get_shape(inputs['points_xyz'])[0]

      cls_loss, loc_loss = model.compute_loss(inputs, preds)

      cls_loss = tf.reduce_sum(cls_loss)
      loc_loss = tf.reduce_sum(loc_loss)
      total_loss += cls_loss.numpy() + loc_loss.numpy()
      total_cls_loss += cls_loss.numpy()
      total_loc_loss += loc_loss.numpy()
      total_example += batch_size

      if total_example % 100 == 0:
        logging.info('finished decoding %d examples', total_example)

    decoded_outputs = {
        'bboxes_pred': tf.concat(bboxes_pred, axis=0),
        'bboxes_pred_score': tf.concat(bboxes_pred_score, axis=0),
        'bboxes_pred_mask': tf.concat(bboxes_pred_mask, axis=0),
        'bboxes': tf.concat(bboxes, axis=0),
        'bboxes_mask': tf.concat(bboxes_mask, axis=0),
        'bboxes_speed': tf.concat(bboxes_speed, axis=0),
    }

    metrics = tf_util.compute_ap(decoded_outputs, FLAGS.class_id)

    val_status = (
        'Val Step: %d / loc_loss = %s, cls_loss = %s.') % (
            current_step, total_loc_loss / total_example,
            total_cls_loss / total_example)

    if eval_summary_writer:
      with eval_summary_writer.as_default():
        tf.summary.scalar(
            'loc_loss', total_loc_loss / total_example, step=current_step)
        tf.summary.scalar(
            'cls_loss', total_cls_loss / total_example, step=current_step)
        tf.summary.scalar(
            'total_example', total_example, step=current_step)
        for metric in metrics:
          for key in metric:
            tf.summary.scalar(key, metric[key], step=current_step)
            metric_status = ('step: %s, %s: %s') % (
                current_step, key, metric[key])
            logging.info(metric_status)

        eval_summary_writer.flush()
    logging.info(val_status)
    if FLAGS.eval_once:
      break


if __name__ == '__main__':
  app.run(main)
