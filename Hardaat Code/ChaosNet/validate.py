import numpy as np

def validate(feat_mat, initial_cond, trajectory_len, epsilon, threshold):
    return (_check_features(feat_mat) and
            _check_trajectory_inputs(initial_cond, threshold, trajectory_len) and
            _check_epsilon(epsilon))

def _check_features(feat_mat):
    if not (isinstance(feat_mat, np.ndarray) and
            feat_mat.dtype == np.float64 and
            feat_mat.ndim == 2):
        print("> ERROR: feat_mat should be 2D array of dtype float64 ...")
        return False
    if feat_mat.min() < 0 or feat_mat.max() > 1:
        print("> ERROR: feat_mat should be scaled between 0 & 1 ...")
        return False
    return True

def _check_trajectory_inputs(init_cond, threshold, trajectory_len):
    if not (isinstance(init_cond, float) and isinstance(threshold, float)):
        print("> ERROR: init_cond & threshold should be of type float ...")
        return False
    if not (0 <= init_cond <= 1 and 0 <= threshold <= 1):
        print("> ERROR: init_condition & threshold cannot be <=0 or >=1 ...")
        return False
    if not (100 <= trajectory_len <= int(1e7) and isinstance(trajectory_len, int)):
        print("> ERROR: length should be an integer between 10^2 & 10^7 ...")
        return False
    return True

def _check_epsilon(epsilon):
    if not (isinstance(epsilon, float) and 1e-5 <= epsilon <= 1.0):
        print("> ERROR: epsilon must be a float between 0.5 and 10^-5")
        return False
    return True