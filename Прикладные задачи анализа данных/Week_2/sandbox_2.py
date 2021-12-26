import pandas as pd
print(pd.__version__)
import numpy as np
print(np.__version__)
import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import numpy as np
from imageio import imread
from imagenet_classes import class_names
import sys
from sklearn.svm import SVC