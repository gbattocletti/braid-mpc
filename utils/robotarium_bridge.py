"""
Helper functions for interfacing with the Robotarium.
"""

# TODO: add an extra robotarium argument to access the real workspace limits (for
# future use with a modified Robotarium class with non-standard workspace)
# Normalize real-world coordinates to [0, 1]

import numpy as np

# Robotarium boundaries
# See: https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/robotarium_abc.py  pylint: disable=line-too-long
BOUNDARIES_POS = [-1.6, -1, 0.4, 1]  # [x_min, y_min, x_max, y_max] square workspace
BOUNDARIES_VEL = [-0.2, -0.2, 0.2, 0.2]  # [x_min, y_min, x_max, y_max]


def real2robotarium(
    x: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray, coords_type: str = "position"
) -> np.ndarray:
    """
    Convert real-world coordinates to Robotarium coordinates.

    Args
        x (np.ndarray): Real-world coordinates of shape (n, 2).
        x_lim (np.ndarray): Limits of the x-axis in the real world, shape (2,).
        y_lim (np.ndarray): Limits of the y-axis in the real world, shape (2,).
        coords_type (str): Type of coordinates to convert ('position' or 'velocity').

    Returns
        np.ndarray: Coordinates in the Robotarium workspace, shape (n, 2).

    Raises
        ValueError: If the type is not 'position' or 'velocity'.
    """
    # Determine boundaries based on the type of coordinates
    match coords_type:
        case "position":
            boundaries = BOUNDARIES_POS
        case "velocity":
            boundaries = BOUNDARIES_VEL
        case _:
            raise ValueError("Invalid coords_type. Must be 'position' or 'velocity'.")

    x_normalized = (x - np.array([x_lim[0], y_lim[0]])) / np.array(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]]
    )

    # Scale to Robotarium workspace size and shift to center
    robotarium_coords = x_normalized * np.array(
        [
            boundaries[2] - boundaries[0],
            boundaries[3] - boundaries[1],
        ]
    ) + np.array([boundaries[0], boundaries[1]])

    return robotarium_coords


def robotarium2real(
    x: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray, coords_type: str = "position"
) -> np.ndarray:
    """
    Convert Robotarium coordinates to real-world coordinates.

    Args
        x (np.ndarray): Robotarium coordinates of shape (n, 2).
        x_lim (np.ndarray): Limits of the x-axis in the real world, shape (2,).
        y_lim (np.ndarray): Limits of the y-axis in the real world, shape (2,).
        coords_type (str): Type of coordinates to convert ('position' or 'velocity').

    Returns
        np.ndarray: Real-world coordinates, shape (n, 2).

    Raises
        ValueError: If the coords_type is not 'position' or 'velocity'.
    """
    # Determine boundaries based on type of coordinates
    match coords_type:
        case "position":
            boundaries = BOUNDARIES_POS
        case "velocity":
            boundaries = BOUNDARIES_VEL
        case _:
            raise ValueError("Invalid coords_type. Must be 'position' or 'velocity'.")

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
