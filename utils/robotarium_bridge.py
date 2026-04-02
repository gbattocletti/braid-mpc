"""
Helper functions for interfacing with the Robotarium.
"""

# TODO: add an extra robotarium argument to access the real workspace limits (for
# future use with a modified Robotarium class with non-standard workspace)
# Normalize real-world coordinates to [0, 1]

import numpy as np


def real2robotarium(x: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray) -> np.ndarray:
    """
    Convert real-world coordinates to Robotarium coordinates.

    Args
        x (np.ndarray): Real-world coordinates of shape (n_x, n).
        x_lim (np.ndarray): Limits of the x-axis in the real world, shape (n_x_pos,).
        y_lim (np.ndarray): Limits of the y-axis in the real world, shape (n_x_pos,).

    Returns
        np.ndarray: Coordinates in the Robotarium workspace, shape (n_x, n).

    Raises
        ValueError: If the type is not 'position' or 'velocity'.
    """
    # Define robotarium boundaries
    # See: https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/robotarium_abc.py  pylint: disable=line-too-long
    boundaries = [-1.6, -1, 0.4, 1]  # [x_min, y_min, x_max, y_max] square workspace

    # Normalize real-world coordinates to [0, 1]
    x_normalized = x[:2]  # only normalize position states
    x_normalized = (
        (x_normalized - np.array([x_lim[0], y_lim[0]]).reshape(-1, 1))
    ) / np.array([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]]).reshape(-1, 1)

    # Scale to Robotarium workspace size and shift to center
    robotarium_coords = x_normalized * np.array(
        [
            boundaries[2] - boundaries[0],
            boundaries[3] - boundaries[1],
        ]
    ).reshape(-1, 1) + np.array([boundaries[0], boundaries[1]]).reshape(-1, 1)

    # Add heading to output vector if needed
    if x.shape[0] == 3:
        robotarium_coords = np.vstack(
            (robotarium_coords, np.reshape(x[2, :], [1, x.shape[1]]))
        )

    return robotarium_coords


def real2robotarium_vel(v: np.ndarray, v_min: np.ndarray, v_max: np.ndarray):
    """
    Convert real-world velocities to Robotarium velocities.

    Args
        v (np.ndarray): Real-world velocities of shape (n, 2).
        v_min (np.ndarray): Minimum limits of the velocity space, shape (2,).
        v_max (np.ndarray): Maximum limits of the velocity space, shape (2,).

    Returns
        np.ndarray: Velocities in the Robotarium workspace, shape (n, 2).
    """
    # Define velocity boundaries
    boundaries = [-0.2, -np.pi, 0.2, np.pi]  # [v_min, omega_min, v_max, omega_max]

    # Normalize real-world velocities to [0, 1]
    v_normalized = (v - v_min) / (v_max - v_min)

    # Scale to Robotarium velocity workspace size and shift to center
    robotarium_vel = v_normalized * np.array(
        [
            boundaries[2] - boundaries[0],
            boundaries[3] - boundaries[1],
        ]
    ) + np.array([boundaries[0], boundaries[1]])

    return robotarium_vel


def robotarium2real(x: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray) -> np.ndarray:
    """
    Convert Robotarium coordinates to real-world coordinates.

    Args
        x (np.ndarray): Robotarium coordinates of shape (n, 2).
        x_lim (np.ndarray): Limits of the x-axis in the real world, shape (2,).
        y_lim (np.ndarray): Limits of the y-axis in the real world, shape (2,).

    Returns
        np.ndarray: Real-world coordinates, shape (n, 2).

    Raises
        ValueError: If the coords_type is not 'position' or 'velocity'.
    """
    # Define robotarium boundaries
    # See: https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/robotarium_abc.py  pylint: disable=line-too-long
    boundaries = [-1.6, -1, 0.4, 1]  # [x_min, y_min, x_max, y_max] square workspace

    # Normalize Robotarium coordinates to [0, 1]
    x_normalized = (x - np.array([boundaries[0], boundaries[1], 0])) / np.array(
        [
            boundaries[2] - boundaries[0],
            boundaries[3] - boundaries[1],
            1,
        ]
    )

    # Scale to real-world size and shift to real-world origin
    real_coords = x_normalized * np.array(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], 1]
    ) + np.array([x_lim[0], y_lim[0], 0])

    return real_coords
