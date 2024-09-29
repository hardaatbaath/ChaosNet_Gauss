import numpy as np
import pandas as pd
import os
import torch

def load_data(directory):
    X_train = pd.read_csv(os.path.join(directory, 'X_train.csv')).values
    y_train = pd.read_csv(os.path.join(directory, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(directory, 'X_test.csv')).values
    y_test = pd.read_csv(os.path.join(directory, 'y_test.csv')).values.ravel()

    return X_train, y_train, X_test, y_test

def save_parameters(params, filename):
    torch.save(params, filename)

def load_parameters(filename):
    return torch.load(filename)