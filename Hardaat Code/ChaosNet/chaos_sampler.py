import numpy as np
from numba import vectorize, float64, njit

@vectorize([float64(float64, float64)])
def _chaotic_map_step(value, *params):
    # Implement your chaotic map here
    # This is a placeholder implementation
    return params[0] * value * (1 - value)

@njit
def _iterate_chaotic_map(params, traj_vec):
    for idx in range(1, len(traj_vec)):
        traj_vec[idx] = _chaotic_map_step(traj_vec[idx - 1], *params)
    return traj_vec

@njit
def _compute_trajectory(init_cond, params, length):
    traj_vec = np.zeros(length, dtype=np.float64)
    traj_vec[0] = init_cond
    return _iterate_chaotic_map(params, traj_vec)

def compute_trajectory(init_cond, params, length, validate=False):
    # Validation can be implemented here if needed
    return _compute_trajectory(init_cond, params, length)

def warmup():
    test_params = np.array([3.8])  # Example parameter for logistic map
    if _compute_trajectory(0.1, test_params, 3)[-1] == np.array([0.4332]):
        print("> Numba JIT warmup successful for chaotic_sampler ...")
    else:
        print("> Numba JIT warmup failed for chaotic_sampler ...")

warmup()
