# import libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics

# from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
import dvc.api

path = 'data/AdSmartABdata.csv'
repo = 'https://github.com/NatnaelSisay/abtest-mlops.git'
data_url = dvc.api.get_url(path=path, repo=repo)
print(data_url)


