import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

CMAP_DEFAULT = 'gray'


def resize_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 128), interpolation=cv2.INTER_LINEAR)
    return image


def prep_image_for_model(image):

    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def load_image(file_path, resize=None, interpolation='linear'):
    """Load image from disk. Output value range: [0,1]."""
    im = cv2.imread(file_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if resize and resize != im.shape[:2]:
        ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
        im = cv2.resize(im, resize, interpolation=ip)
    return im.astype(np.float32) / 255.0


def gray2rgb(im, cmap=CMAP_DEFAULT):
  cmap = plt.get_cmap(cmap)
  result_img = cmap(im.astype(np.float32))
  if result_img.shape[2] > 3:
    result_img = np.delete(result_img, 3, 2)
  return result_img


def normalize_depth(depth, normalizer=None, pc=95):
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  norm_disp = np.clip(disp, 0, 1)
  return norm_disp


def depth_to_rgb(depth, crop_percent=0, cmap=CMAP_DEFAULT):
  norm_disp = normalize_depth(depth)
  disp = gray2rgb(norm_disp, cmap='plasma')
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)