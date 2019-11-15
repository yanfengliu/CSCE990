import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


tf.train.list_variables(tf.train.latest_checkpoint('./pre_trained/'))