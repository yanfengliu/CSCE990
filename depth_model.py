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


def process_video(inference_model, sess, FLAGS):
    width = 416
    height = 128
    count = 0
    total = 2000
    video_path = f'video/seattle.mp4'
    print(f'Processing {video_path}')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30000)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened() and count < total):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'output/color/{count}.png', frame)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32) / 255.0
            depth = inference_model.inference_depth(frame, sess)
            depth = np.squeeze(depth)
            depth = util.normalize_depth_for_display(depth)
            depth = depth * 255
            cv2.imwrite(f'output/depth/{count}.png', depth)
            print(count)
            count += 1
        else: 
            break
    cap.release()