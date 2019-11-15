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
from params import Params
from depth_model import *


FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'output'
FLAGS.checkpoint_path = 'pre_trained/model.ckpt'
FLAGS.img_height = 128
FLAGS.img_width = 416

inference_model, sess = init_inference_model(FLAGS)
process_folder(inference_model, sess, FLAGS)
