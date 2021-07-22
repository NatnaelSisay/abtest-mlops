import pandas as pd
import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    data = pd.read_csv('/data/AdSmartABdata.csv')
    # Log a parameter (key-value pair)
    log_param("Number of rows", data.shape[0])
    log_param("Number of Columns", data.shape[1])

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")