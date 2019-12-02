import io
import math
import os
import random
import time

import numpy as np
import cv2

import model
import util
from params import Params
from depth_model import *


FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'output'
FLAGS.checkpoint_path = 'pre_trained/model'
FLAGS.img_height = 128
FLAGS.img_width = 416

# inference_model, sess = init_inference_model(FLAGS)
# process_video(inference_model, sess, FLAGS)

root_dir ='C:/Users/yliu60/Documents/GitHub/CSCE990/output'
out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (416*2, 128))
count = 0
filelist = os.listdir(os.path.join(root_dir, 'color'))
filelist.sort(key=lambda f: int(f.split('.')[0]))
for filename in filelist:
    if count < 2000:
        color = cv2.imread(os.path.join(root_dir, 'color', filename))
        # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        depth = cv2.imread(os.path.join(root_dir, 'depth', filename))
        # depth = cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)
        board = np.zeros((128, 416*2, 3))
        board[:, :416, :] = color
        board[:, 416:, :] = depth
        board = board * 255
        board = board.astype(np.uint8)
        # need to be BGR format
        out.write(board)
        count += 1
        print(count)
    else:
        break
out.release()
