import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CMAP_DEFAULT = 'gray'


def prep_image_for_model(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 128), interpolation=cv2.INTER_LINEAR)
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


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap=CMAP_DEFAULT):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.

  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp
