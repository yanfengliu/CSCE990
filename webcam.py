import cv2
import numpy as np

import depth_model
import util
from params import Params

FLAGS = Params()
FLAGS.input_dir = 'input'
FLAGS.output_dir = 'ouptut'
FLAGS.checkpoint_path = 'pre_trained/model'
FLAGS.img_height = 128
FLAGS.img_width = 416


cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    print('Reading failed')

inference_model, sess = depth_model.init_inference_model(FLAGS)

while rval:
    rval, image = vc.read()
    image = util.prep_image_for_model(image)
    depth = inference_model.inference_depth(image, sess)
    depth = np.squeeze(depth)
    depth = util.normalize_depth_for_display(depth)
    cv2.imshow("preview", depth)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
