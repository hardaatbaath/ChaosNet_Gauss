import numpy as np
import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.drop('target', axis=1).values
    y = data['target'].values
    return X, y

def save_parameters(params, filename):
    np.save(filename, params)

def load_parameters(filename):
    return np.load(filename)