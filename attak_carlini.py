
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'max_iterations', 30, 'Number of iterations to perform gradient descent.')

tf.flags.DEFINE_float(
    'learning_rate', 1e-1, 'Larger values converge faster to less accurate results.')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')

class InceptionModel:
  def __init__(self, num_labels):
    self.num_labels = num_labels

  def predict(self, img):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
        img, num_classes=self.num_labels, is_training=False)
        output = end_points['Predictions']
    return output

def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  with tf.Graph().as_default():
    # Prepare graph
    model = InceptionModel(num_classes)

    # the variable we're going to optimize over
    dw = tf.Variable(tf.zeros(batch_shape), name='Modifier')

    # these are variables to be more efficient in sending data to tf
    x = tf.Variable(tf.zeros(batch_shape), name='Image')
    y = tf.Variable(tf.zeros([FLAGS.batch_size],dtype=tf.int32), name='Label')

    # and here's what we use to assign them
    x_input = tf.placeholder(tf.float32, batch_shape)
    y_input = tf.placeholder(tf.int32, (FLAGS.batch_size))

    # constants: eps, clip_value_min, and clip_value_max
    e = tf.constant(eps, dtype=tf.float32)
    clip_value_min = tf.constant(-1.0, dtype=tf.float32)
    clip_value_max = tf.constant(1.0, dtype=tf.float32)

    x_min = tf.clip_by_value(x - e, clip_value_min, clip_value_max)
    x_max = tf.clip_by_value(x + e, clip_value_min, clip_value_max)
    x_width = (x_max - x_min) * 0.5

    # x must be in [x_min, x_max] interval.
    # the scaled x is bounded from -1 to 1.
    x_scaled = (x-x_min) / x_width - 1.0
    w = tf.atanh(x_scaled*0.999999)

    # the resulting image, tanh'd to keep bounded from x_min to x_max
    x_new = (tf.tanh(dw + w) + 1.0) * x_width + x_min

    # prediction of the model
    output = model.predict(x_new)

    # sum up the loss
    y1 = tf.one_hot(y, depth=num_classes, on_value=1.0, off_value=0.0, axis=-1)
    loss = -tf.reduce_sum(tf.multiply(y1,tf.log(output)))

    # Setup the adam optimizer and keep track of variables we're creating
    start_vars = set(v.name for v in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train = optimizer.minimize(loss, var_list=[dw])
    end_vars = tf.global_variables()
    new_vars = [v for v in end_vars if v.name not in start_vars]

    # these are the variables to initialize when we run
    set_vars = [
        x.assign(x_input),
        y.assign(y_input)
        ]

    init_vars = tf.variables_initializer(var_list=[dw]+new_vars)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))
    session_creator = tf.train.ChiefSessionCreator(
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        target_class_for_batch = (
            [all_images_taget_class[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames)))

        # completely reset adam's internal state.
        sess.run(init_vars)

        # set the variables so that we don't have to send them over again
        sess.run(set_vars, {x_input: images,
                            y_input: target_class_for_batch})

        for iteration in range(FLAGS.max_iterations):
            # perform the attack
            _, l, adv_images = sess.run([train, loss, x_new])
            print(iteration, l)

        save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
