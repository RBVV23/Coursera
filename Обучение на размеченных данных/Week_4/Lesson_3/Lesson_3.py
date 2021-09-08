from sklearn import ensemble, model_selection, metrics
import numpy as np
import pandas as pd
import xgboost as xgb
print(xgb.__version__)

bioresponce = pd.read_csv('bioresponse.csv', header=0, sep=',')
