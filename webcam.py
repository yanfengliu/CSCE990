import datetime
import os
import time

import cv2
import numpy as np

import depth_model
import util
from params import Params

alpha = 0.5
height = 128
width = 416
h_min = int(0.33 * height)
h_max = int(0.67 * height)
xy_pairs = [
    [(  0, h_min + height), ( 83, h_max + height)],
    [( 83, h_min + height), (166, h_max + height)],
    [(166, h_min + height), (249, h_max + height)],
    [(249, h_min + height), (332, h_max + height)],
    [(332, h_min + height), (415, h_max + height)]
]
xy_pairs_h = [
    [(  0, h_min + height), ( 83, h_max + height)],
    [( 83, h_min + height), (166, h_max + height)],
    [(166, h_min + height), (249, h_max + height)],
    [(249, h_min + height), (332, h_max + height)],
    [(332, h_min + height), (415, h_max + height)]
]
highlight_color = (255, 255, 255)

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


def draw_regions(board):
    for xy_pair in xy_pairs_h:
        x1y1 = xy_pair[0]
        x2y2 = xy_pair[1]
        cv2.rectangle(board, x1y1, x2y2, highlight_color, 2)
    return board


def draw_direction(board, direction):
    overlay = board.copy()
    xy_pair = xy_pairs[direction]
    x1y1 = xy_pair[0]
    x2y2 = xy_pair[1]
    overlay = cv2.rectangle(overlay, x1y1, x2y2, highlight_color, -1)
    output = cv2.addWeighted(overlay, alpha, board, 1 - alpha, 0)
    return output


def get_dists(norm_depth):
    dists = []
    for i in range(5):
        w_min = int(width * i / 5)
        w_max = int(width * (i+1) / 5)
        dist = np.mean(norm_depth[h_min:h_max, w_min:w_max])
        dists.append(dist)
    return dists


def get_direction(dists):
    # right now there is no laser scanner for true dists so
    # obstacle avoidance is very basic
    return np.argmin(dists)


ts = time.time()
time_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
img_dir = f'img/{time_str}'
util.mkdir_if_missing(img_dir)
count = 0
while rval:
    board = np.zeros((2*height, width, 3), dtype=np.uint8)
    rval, raw_image = vc.read()
    raw_image = util.resize_img(raw_image)
    image = util.prep_image_for_model(raw_image)
    depth = inference_model.inference_depth(image, sess)
    depth = np.squeeze(depth)
    norm_depth = util.normalize_depth(depth)
    depth_rgb = util.depth_to_rgb(depth)
    depth_rgb *= 255
    depth_rgb = depth_rgb.astype(np.uint8)
    dists = get_dists(norm_depth)
    direction = get_direction(dists)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
    board[:height, :, :] = raw_image
    board[height:, :, :] = depth_bgr
    board = draw_regions(board)
    board = draw_direction(board, direction)
    cv2.imshow("preview", board)
    cv2.imwrite(os.path.join(img_dir, f'{count}.png'), board)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    count += 1
cv2.destroyWindow("preview")
