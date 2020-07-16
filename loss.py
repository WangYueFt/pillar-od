"""Loss function."""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()


def sigmoid_cross_entropy_focal_loss(logits, labels, alpha=0.25, gamma=2.0):
  """Focal loss for binary (sigmoid) logistic loss."""
  # The numerically-stable way to compute
  #  log(p) for positives;
  #  log(1 - p) for negatives.
  labels = tf.cast(labels, logits.dtype)
  labels = tf.reshape(labels, logits.shape)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  if gamma is not None and gamma != 0:
    # The modulating factor. Note that
    inner = tf.sigmoid(logits * (1 - labels * 2))
    loss *= tf.pow(inner, gamma)

  if alpha is not None:
    # [1] Eq (3)
    loss *= (alpha * labels + (1 - alpha) * (1 - labels))

  loss = tf.reduce_sum(loss, axis=-1)
  return loss


def smooth_l1_loss(predictions, labels, sigma=3.0):
  """Smooth L1 loss."""
  predictions = tf.cast(predictions, tf.dtypes.float32)
  labels = tf.cast(labels, tf.dtypes.float32)
  diff = tf.abs(predictions - labels)
  losses = tf.where(diff < 1.0 / sigma / sigma,
                    0.5 * sigma * sigma * diff * diff,
                    diff - 0.5 / sigma / sigma)
  return losses
