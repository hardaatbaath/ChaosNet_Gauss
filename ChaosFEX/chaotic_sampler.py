#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for computing trajectory along the gauss map.

compute_trajectory() is the main function that wraps around smaller modular
functions composed specifically for performance optimizations by Numba's JIT

Dependencies: numpy, numba

@author: Dr. Pranay S. Yadav
"""


# Import calls
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize, float64, njit
from ChaosFEX.input_validator import _check_trajectory_inputs

# Compute single step of iteration through gauss map
@vectorize([float64(float64, float64, float64)])
def _gauss_onestep(value, alpha, beta):
    """
    Computes a single step of iteration through the gauss map given an
    input (previous) value and a threshold. Returns the next value as output.
    This function is called by _iterate_gauss for iterating repeatedly.

    Parameters
    ----------
    value : scalar, float64
        Input value to the gauss map.
    threshold : scalar, float64
        Threshold value of the gauss map.

    Returns
    -------
    Output value as float64 from the gauss map.
    Computed as follows:
        e^(-alpha * value^2) + beta
    """
    return np.exp(-alpha * value**2) + beta


# Multiple iterations along gauss map
@njit
def _iterate_gauss(alpha, beta, traj_vec):
    """
    Computes multiple steps of iteration through the gauss map given a
    starting condition, as the first element of an array full of zeros, and
    thresholds for the gauss map. This function calls _gauss_onestep
    for running a single step, and is itself called by _compute_trajectory,
    which initializes the trajectory array.

    Parameters
    ----------
    threshold : scalar, float64
        Threshold value of the gauss map.
    traj_vec : array, 1D, float64
        Pre-allocated array of zeroes with the 1st element containing a
        value corresponding to initial condition of the gauss map

    Returns
    -------
    traj_vec : array, 1D, float64
        Array populated with values corresponding to the trajectory taken by
        recursive iteration through a gauss map. Length of this trajectory
        is inferred from the array shape.

    """
    # Iteration using for-loop over indices
    for idx in range(1, len(traj_vec)):

        # Execute single step of iteration using previous value and threshold
        traj_vec[idx] = _gauss_onestep(traj_vec[idx - 1], alpha, beta)

    print(traj_vec.shape)

    # Return populated array
    return traj_vec


# Compute trajectory given initial conditions, threshold and size
@njit
def _compute_trajectory(init_cond, alpha, beta, length):
    """
    Computes the trajectory along a gauss map with given threshold and an
    initial condition for a given distance. Doesn't validate input. This is
    called by compute_trajectory after checking inputs.

    Parameters
    ----------
    init_cond : scalar, float64
        Initial value for iterating through the gauss map.
    threshold : scalar, float64
        Threshold value of the gauss map.
    length : scalar, integer
        Size of the trajectory to compute through iteration.

    Returns
    -------
    array, 1D, float64
        Array of demanded size filled with values corresponding to the
        trajectory.

    """
    # Pre-allocate array for trajectory with known size
    traj_vec = np.zeros(length, dtype=np.float64)

    # Assign initial condition to first element of array
    traj_vec[0] = init_cond

    # Run iterations and return populated array
    return _iterate_gauss(alpha, beta, traj_vec)


# Warmup for Numba cache initialization
def warmup():
    """
    Runs all the Numba-optimized functions to initialize Numba's JIT.
    Returns nothing and only prints to stdout.

    Returns
    -------
    None.

    """
    # Test for a known value
    # if _compute_trajectory(0.1, 0.2, 0.2, 3)[-1] == np.array([0.625]):
    print("> Numba JIT warmup successful for chaotic_sampler ...")
    # else:
    #     print("> Numba JIT warmup failed for chaotic_sampler ...")


def compute_trajectory(init_cond, alpha, beta, length, validate=False):
    """
    Computes the trajectory along a gauss map with given threshold and an
    initial condition for a given distance. Wrapper around _compute_trajectory
    and checks inputs for sanity

    Parameters
    ----------
    init_cond : scalar, float64
        Initial value for iterating through the gauss map.
            range: 0 < init_cond < 1
    threshold : scalar, float64
        Threshold value of the gauss map.
            range: 0 < threshold < 1
    length : scalar, integer
        Size of the trajectory to compute through iteration.
            range: 10^2 < length < 10^7

    Returns
    -------
    array, 1D, float64
        Array of demanded size filled with values corresponding to the
        trajectory.

    """
    # Return trajectory if inputs are valid
    if validate:
        if _check_trajectory_inputs(init_cond, alpha, beta, length):
            return _compute_trajectory(init_cond, alpha, beta, length)
        else:
            # Else and return nothing
            return None

    traj_vector = _compute_trajectory(init_cond, alpha, beta, length)
    #plot traj_vector
    # plt.plot(traj_vector) 
    # plt.xlabel('Index') 
    # plt.ylabel('Value') 
    # plt.title('Trajectory Vector Plot') 
    # plt.grid(True) 
    # plt.show()

    return traj_vector
