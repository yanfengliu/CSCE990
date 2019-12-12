import cv2
import numpy as np

import depth_model
import util
from params import Params

height = 128
width = 416

FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'ouptut'
FLAGS.checkpoint_path = 'pre_trained/model.ckpt'
FLAGS.img_height = height
FLAGS.img_width = width


cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    print('Reading failed')

inference_model, sess = depth_model.init_inference_model(FLAGS)

while rval:
    board = np.zeros((2*height, width))
    rval, image = vc.read()
    image = util.prep_image_for_model(image)
    depth = inference_model.inference_depth(image, sess)
    depth = np.squeeze(depth)
    depth = util.normalize_depth_for_display(depth)
    board[:height, :, :] = image
    board[height:, :, :] = depth
    cv2.imshow("preview", board)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
