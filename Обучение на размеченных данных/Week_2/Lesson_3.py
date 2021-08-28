from matplotlib.colors import ListedColormap
from sklearn import model_selection, datasets, linear_model, metrics

import matplotlib.pyplot as plt
import numpy as np

blobs = datasets.make_blobs(centers=2, cluster_std=5.5, random_state=1)