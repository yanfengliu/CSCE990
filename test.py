import io
import math
import os
import random
import time

import numpy as np
import cv2

import util
from params import Params
from depth_model import *


FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'output'
FLAGS.checkpoint_path = 'pre_trained/model'
FLAGS.img_height = 128
FLAGS.img_width = 416

inference_model, sess = init_inference_model(FLAGS)
input_dir = 'C:/Users/yliu60/Documents/GitHub/CSCE990/videos'
output_dir = 'C:/Users/yliu60/Documents/GitHub/CSCE990/processed_videos'

video_path_list = os.listdir(input_dir)
for video_name in video_path_list:
    video_path = os.path.join(input_dir, video_name)
    clean_video_name = video_name.split('.')[0]
    save_dir = os.path.join('C:/Users/yliu60/Documents/GitHub/CSCE990/frames', clean_video_name)
    color_save_dir = os.path.join(save_dir, 'color')
    depth_save_dir = os.path.join(save_dir, 'depth')
    process_video(video_path, save_dir, inference_model, sess, FLAGS)
    output_path = os.path.join(output_dir, f'{clean_video_name}.avi')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (416, 128*2))
    count = 0
    filelist = os.listdir(color_save_dir)
    filelist.sort(key=lambda f: int(f.split('.')[0]))
    for filename in filelist:
        if count < 2000:
            board = np.zeros((128*2, 416, 3))
            image = cv2.imread(os.path.join(color_save_dir, filename))
            depth = cv2.imread(os.path.join(depth_save_dir, filename))
            board[:128, :, :] = image
            board[128:, :, :] = depth
            board = board.astype(np.uint8)
            out.write(board)
            count += 1
            print(count)
        else:
            break
    out.release()
