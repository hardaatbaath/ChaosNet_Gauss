import numpy as np
import numba as nb
from .chaos_sampler import ChaosSampler

class FeatureExtractor:
    @staticmethod
    def _compare(value1, value2, value3):
        return abs(value1 - value2) < value3

    @staticmethod
    def _compute_match_idx(value, array, epsilon):
        for idx in range(len(array)):
            if FeatureExtractor._compare(value, array[idx], epsilon):
                return idx
        return len(array)

    @staticmethod
    def _compute_energy(path):
        return path @ path

    @staticmethod
    def _compute_ttss_entropy(path, threshold):
        """
        Adjust this function to use the threshold array. We assume the first threshold element
        is used, but this can be customized to use multiple thresholds if needed.
        """
        prob = np.count_nonzero(path > threshold[0]) / len(path)
        return np.array([prob, -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)])

    @staticmethod
    def _compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat):
        """
        Update this function to accept the threshold array and pass it correctly during processing.
        """
        for i in nb.prange(feat_mat.shape[0]):
            for j in nb.prange(feat_mat.shape[1]):
                idx = FeatureExtractor._compute_match_idx(feat_mat[i, j], trajectory, epsilon)
                meas_mat[i, j, 2] = idx
                if idx != 0:
                    path = trajectory[:idx]
                    meas_mat[i, j, 1] = FeatureExtractor._compute_energy(path)
                    ttss_entropy = FeatureExtractor._compute_ttss_entropy(path, threshold)
                    meas_mat[i, j, 0] = ttss_entropy[0]
                    meas_mat[i, j, 3] = ttss_entropy[1]
        return meas_mat

    @staticmethod
    def transform(feat_mat, initial_cond, trajectory_len, epsilon, threshold):
        """
        Modify this function to take a threshold array instead of a single value.
        """
        from .validate import validate
        if not validate(feat_mat, initial_cond, trajectory_len, epsilon, float(threshold[0])):  # Validate only first threshold element for simplicity
            return None

        dimx, dimy = feat_mat.shape
        meas_mat = np.zeros([dimx, dimy, 4])
        trajectory = ChaosSampler.compute_trajectory(initial_cond, threshold, trajectory_len)
        out = FeatureExtractor._compute_measures(feat_mat, trajectory, epsilon, threshold, meas_mat)
        out[:, :, 3] = np.nan_to_num(out[:, :, 3])
        out = out.transpose([0, 2, 1]).reshape([dimx, dimy * 4])
        return out

    @staticmethod
    def warmup():
        feat_mat = np.array([[0.1, 0.2], [0.3, 0.4]])
        out = FeatureExtractor.transform(feat_mat, initial_cond=0.1, trajectory_len=100, epsilon=0.01, threshold=[0.2])
        if out.shape == (2, 8) and out[0, 5] == 12:
            print("> Numba JIT warmup successful for transform ...")
        else:
            print("> Numba JIT warmup failed for transform ...")

FeatureExtractor.warmup()
