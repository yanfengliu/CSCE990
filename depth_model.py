import io
import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import util
import model


def init_inference_model(FLAGS):
    inference_model = model.Model(
        is_training=False,
        batch_size=1,
        img_height=FLAGS.img_height,
        img_width=FLAGS.img_width)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, FLAGS.checkpoint_path)
    return inference_model, sess


def process_folder(inference_model, sess, FLAGS):
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # input of shape [self.batch_size, self.img_height, self.img_width, 3]
    image_list = os.listdir(FLAGS.input_dir)
    for image_name in image_list:
        image_path = os.path.join(FLAGS.input_dir, image_name)
        image = util.load_image(image_path)
        image = np.expand_dims(image, axis=0)
        depth = inference_model.inference_depth(image, sess)
        depth = util.normalize_depth_for_display(depth)
        cv2.imwrite(os.path.join(FLAGS.output_dir, f'depth-{image_name}', depth))