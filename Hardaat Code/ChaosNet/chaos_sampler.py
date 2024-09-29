import numpy as np
from numba import vectorize, float64, njit

class ChaosSampler:
    @staticmethod
    def _skewtent_onestep(value, threshold):
        """
        Adjusted to allow the threshold array to be used flexibly.
        """
        if value < threshold[0]:
            return value / threshold[0]
        return (1 - value) / (1 - threshold[0])

    @staticmethod
    def _iterate_skewtent(threshold, traj_vec):
        for idx in range(1, len(traj_vec)):
            traj_vec[idx] = ChaosSampler._skewtent_onestep(traj_vec[idx - 1], threshold)
        return traj_vec

    @staticmethod
    def _compute_trajectory(init_cond, threshold, length):
        traj_vec = np.zeros(length, dtype=np.float64)
        traj_vec[0] = init_cond
        return ChaosSampler._iterate_skewtent(threshold, traj_vec)

    @staticmethod
    def compute_trajectory(init_cond, threshold, length, validate=False):
        if validate:
            from .validate import _check_trajectory_inputs
            if not _check_trajectory_inputs(init_cond, threshold, length):
                print("> ERROR: Invalid inputs for compute_trajectory ...")
                return None
        return ChaosSampler._compute_trajectory(init_cond, threshold, length)

    @staticmethod
    def warmup():
        if ChaosSampler._compute_trajectory(0.1, [0.2], 3)[-1] == np.array([0.625]):
            print("> Numba JIT warmup successful for chaotic_sampler ...")
        else:
            print("> Numba JIT warmup failed for chaotic_sampler ...")

ChaosSampler.warmup()
