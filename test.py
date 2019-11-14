import io
import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import model
import util
from depth_model import *


class Params():
    def __init__(self):
        pass

FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'ouptut'
FLAGS.checkpoint_path = 'model_zoo'
FLAGS.img_height = 128
FLAGS.img_width = 416

inference_model, sess = init_inference_model(FLAGS)
process_folder(inference_model, sess, FLAGS)
