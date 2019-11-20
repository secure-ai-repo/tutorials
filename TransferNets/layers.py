from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import bias_add
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected as fc
from tensorflow.contrib.layers import l2_regularizer as l2
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import separable_conv2d
from tensorflow.contrib.layers import variance_scaling_initializer


