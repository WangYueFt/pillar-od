"""Config flags."""
from absl import flags


def define_flags():
  """Add training flags."""

  flags.DEFINE_string('master', 'local', 'Location of the session.')

  flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU.')

  flags.DEFINE_string('model_dir', '/tmp/results/ped_eval/',
                      'training directory root')

  # Optimizer
  flags.DEFINE_float('lr', 3e-3, 'learning rate')
  flags.DEFINE_integer('epochs', 75, 'number of batches to train on')

  # Dataset
  flags.DEFINE_integer('batch_size', 1,
                       'batch size for training')
  flags.DEFINE_integer('test_batch_size', 2,
                       'batch size for testing')
  flags.DEFINE_integer('cycle_length', 128,
                       'number of parallel file readers')
  flags.DEFINE_integer('num_parallel_calls', 128,
                       'number of parallel dataloader threads')
  flags.DEFINE_integer('shuffle_buffer_size', 1024,
                       'buffer size for shuffling data')
  flags.DEFINE_float('percentile', 1.00, 'percentile of validation data.')
  flags.DEFINE_string('data_path', '/home/yuewang/data/waymo/processed', 'data path')

  # Task specific params
  flags.DEFINE_integer('max_num_points', 245760,
                       'maximum number of lidar points')
  flags.DEFINE_integer('max_num_bboxes', 200,
                       'maximum number of bounding bboxes')
  flags.DEFINE_integer('class_id', 2, 'class id (car=1, pedestrian=2)')
  flags.DEFINE_integer('difficulty', 1, 'difficulty level (1 or 2)')

  flags.DEFINE_integer('pillar_map_size', 512, 
                       'birds-eye view pillar size (256 for car, 512 for pedestrian)')
  flags.DEFINE_float('pillar_map_range', 75.2, 'birds-eye view detection range')

  flags.DEFINE_string('norm_type', 'sync_batch_norm',
                      'normalization type to use')
  flags.DEFINE_string('act_type', 'relu',
                      'activation type to use')
  flags.DEFINE_float('nms_iou_threshold', 0.2,
                     'nms ios threshold (0.7 for car, 0.2 for pedestrian)')
  flags.DEFINE_float('nms_score_threshold', 0.0, 'prediction score threshold')
  flags.DEFINE_integer('max_nms_boxes', 200,
                       'maximum number of bounding boxes to keep after NMS')
  flags.DEFINE_bool('use_oriented_per_class_nms', True,
                    'whether to use oriented NMS')

  # For evaluation
  flags.DEFINE_bool('eval_once', True, 'eval once or forever (during training)')
  flags.DEFINE_string('ckpt_path', '/home/yuewang/data/waymo_pretrained_model/eccv/ped/ped',
                      'checkpoint path')
  return flags.FLAGS
