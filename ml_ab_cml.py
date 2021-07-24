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

from urllib.parse import urlparse

path = 'data/AdSmartABdata.csv'
repo = 'https://github.com/NatnaelSisay/abtest-mlops.git'
data_url = dvc.api.get_url(path=path, repo=repo)


# reading data
ad_df = pd.read_csv(data_url)
df = ad_df.copy()
df['conversion'] = df.yes

# drop unecessary columns
df.drop(['yes', 'no', 'auction_id'], axis=1, inplace=True)

# label encode columns
lb_encode = LabelEncoder()
df['experiment'] = lb_encode.fit_transform(df['experiment'])
df['date'] = lb_encode.fit_transform(df['date'])
df['device_make'] = lb_encode.fit_transform(df['device_make'])
df['browser'] = lb_encode.fit_transform(df['browser'])

# Scale data
scaler = MinMaxScaler()
scalled = scaler.fit_transform(df)
scalled_df = pd.DataFrame(data = scalled, columns=df.columns)

# Split to dependent and independet columns
data_x = scalled_df.loc[:, df.columns != 'conversion']
data_y = scalled_df['conversion']

#### Split the data into 70% training, 20% validation, and 10% test sets. 
X_train, X_test, y_train, y_test\
    = train_test_split(data_x, data_y, test_size=0.3, random_state=1)

X_val, X_test, y_val, y_test\
    = train_test_split(X_test, y_test, test_size=0.10, random_state=1)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#### Train Models
if __name__ == '__main__':
  lr = LogisticRegression()
  lr.fit(X_train, y_train)

  dt = DecisionTreeClassifier()
  dt.fit(X_train, y_train)

  #### Mean Score after cross-validation with kfold of 5
  lr_results = cross_val_score(lr, X_train, y_train, cv=5)
  dt_result = cross_val_score(dt, X_train, y_train, cv=5)

  lr_accuracy = round(lr_results.mean() * 100,2)
  dt_accuracy = round(dt_result.mean() * 100,2)
  
  
  ### Loss Pridiction
  lr_predict = lr.predict(X_val)
  dt_predict = dt.predict(X_val)
  lr_loss = round(mean_squared_error(y_val,lr_predict) * 100, 2)
  dt_loss = round(mean_squared_error(y_val, dt_predict) * 100, 2)
  
  (rmse, mae, r2) = eval_metrics(y_val, lr_predict)
  (dt_rmse, dt_mae, dt_r2) = eval_metrics(y_val, lr_predict)

  with open('results.txt', 'w') as result:
      result.write('Logistic Regretion\n')
      result.write(f'Model Accuracy : {lr_accuracy} %\n')
      result.write(f'Model Loss : {lr_loss} %\n')
      result.write(f"rmse:{rmse}\tmae:{mae}\tr2:{r2}\n")

      result.write('\n')
      result.write('Decision Tree\n')
      result.write(f'Model Accuracy : {dt_accuracy} %\n')
      result.write(f'Model Loss : {dt_loss} %\n')
      result.write(f'rmse:{dt_rmse}\tmae:{dt_mae}\tr2:{dt_r2}\n')

