import numpy as np
import numba as nb
from .chaotic_sampler import compute_trajectory

@nb.vectorize([nb.boolean(nb.float64, nb.float64, nb.float64)])
def _compare(value1, value2, value3):
    return abs(value1 - value2) < value3

@nb.njit
def _compute_match_idx(value, array, epsilon):
    length = len(array)
    for idx in range(length):
        if _compare(value, array[idx], epsilon):
            return idx
    return length

@nb.njit
def _compute_energy(path):
    return path @ path

@nb.njit
def _compute_ttss_entropy(path, threshold):
    prob = np.count_nonzero(path > threshold) / len(path)
    return np.array([prob, -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)])

@nb.njit(parallel=True)
def _compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat):
    for i in nb.prange(feat_mat.shape[0]):
        for j in nb.prange(feat_mat.shape[1]):
            idx = _compute_match_idx(feat_mat[i, j], trajectory, epsilon)
            meas_mat[i, j, 2] = idx
            if idx != 0:
                path = trajectory[:idx]
                meas_mat[i, j, 1] = _compute_energy(path)
                ttss_entropy = _compute_ttss_entropy(path, threshold)
                meas_mat[i, j, 0] = ttss_entropy[0]
                meas_mat[i, j, 3] = ttss_entropy[1]
    return meas_mat

def transform(feat_mat, initial_cond, trajectory_len, epsilon, params):
    dimx, dimy = feat_mat.shape
    meas_mat = np.zeros([dimx, dimy, 4])
    trajectory = compute_trajectory(initial_cond, params, trajectory_len)
    threshold = params[0]  # Assuming first parameter is threshold
    out = _compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat)
    out[:, :, 3] = np.nan_to_num(out[:, :, 3])
    out = out.transpose([0, 2, 1]).reshape([dimx, dimy * 4])
    return out

def warmup():
    feat_mat = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = transform(feat_mat, initial_cond=0.1, trajectory_len=100, epsilon=0.01, params=np.array([0.2]))
    if out.shape == (2, 8) and out[0, 5] == 12:
        print("> Numba JIT warmup successful for transform ...")
    else:
        print("> Numba JIT warmup failed for transform ...")

warmup()