import numpy as np
from scipy.optimize import minimize
import torch
from sklearn.model_selection import KFold

def compute_trajectory(initial_cond, threshold, length):
    trajectory = np.zeros(length)
    trajectory[0] = initial_cond
    for i in range(1, length):
        x = trajectory[i-1]
        if x < threshold:
            trajectory[i] = x / threshold
        else:
            trajectory[i] = (1 - x) / (1 - threshold)
    return trajectory

def extract_features(feature_matrix, trajectory, epsilon):
    features = []
    for sample in feature_matrix:
        sample_features = []
        for value in sample:
            idx = np.argmin(np.abs(trajectory - value))
            ttss = np.mean(trajectory[:idx] > 0.5)  # Time To Settle State
            energy = np.sum(trajectory[:idx]**2)
            tt = idx  # Traversal Time
            entropy = -np.sum(trajectory[:idx] * np.log2(trajectory[:idx] + 1e-10))
            sample_features.extend([ttss, energy, tt, entropy])
        features.append(sample_features)
    return np.array(features)

def objective_function(params, feature_matrix):
    initial_cond, epsilon, threshold = params
    if not all(0 <= p <= 1 for p in params):
        return np.inf
    
    trajectory = compute_trajectory(initial_cond, threshold, 1000)
    features = extract_features(feature_matrix, trajectory, epsilon)
    
    # Use a simple metric for optimization
    return -np.sum(features)

def optimize_chaotic_params(feature_matrix):
    initial_guess = [0.5, 0.01, 0.5]  # initial_cond, epsilon, threshold
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    result = minimize(
        lambda x: objective_function(x, feature_matrix),
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return result.x

def k_fold_split(data, labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(data):
        yield data[train_index], labels[train_index], data[val_index], labels[val_index]
