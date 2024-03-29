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
import matplotlib.pyplot as plt 
import model
import cv2


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
        image = util.load_image(image_path, resize=(416, 128))
        image = np.expand_dims(image, axis=0)
        depth = inference_model.inference_depth(image, sess)
        depth = np.squeeze(depth)
        depth = util.normalize_depth_for_display(depth)
        cv2.imwrite(os.path.join(FLAGS.output_dir, f'depth-{image_name}'), depth)


def process_video(video_path, save_dir, inference_model, sess, FLAGS):
    width = 416
    height = 128
    count = 0
    total = 2000
    color_save_dir = os.path.join(save_dir, 'color')
    depth_save_dir = os.path.join(save_dir, 'depth')
    util.mkdir_if_missing(color_save_dir)
    util.mkdir_if_missing(depth_save_dir)
    print(f'Processing {video_path}')
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened() and count < total):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(color_save_dir, f'{count}.png'), frame)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32) / 255.0
            depth = inference_model.inference_depth(frame, sess)
            depth = np.squeeze(depth)
            depth = util.normalize_depth_for_display(depth)
            depth = depth * 255
            depth = depth.astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(depth_save_dir, f'{count}.png'), depth)
            print(count)
            count += 1
        else: 
            break
    cap.release()