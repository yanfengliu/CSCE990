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

start = time.time()
inference_model, sess = init_inference_model(FLAGS)
for i in range(1000):
    print(i)
    frame = np.random.rand(1, 128, 416, 3)
    depth = inference_model.inference_depth(frame, sess)
end = time.time()
print(f'Used {end-start} seconds for 1000 frames.')