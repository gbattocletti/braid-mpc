"""
Utility functions to convert between different representations of topological
specifications, such as braids, paths, sequences of grids, and winding numbers.
"""

import numpy as np


def grids2paths(grids: np.ndarray) -> np.ndarray:
    """
    Convert a list of grids to a list of paths.

    Args:
        grids (np.ndarray): A 3D array of shape (n, m, m) representing n time steps of
            m x m grids. Each grid represents the relative positions of m agents at a
            given time step. Each row and column of the grid can contain at most one
            non-zero entry corresponding to an index i

    Returns:
        np.ndarray: A 3D array of shape (n, 2, m) representing n time steps of m agents'
            positions in 2D space. Each entry paths[t, :, i] gives the (x, y)
            coordinates of agent i at time step t.

    Raises:
        TypeError: If the input 'grids' is not a numpy array.
        ValueError: If the input 'grids' does not have the correct shape.
        ValueError: If any agent is missing or appears multiple times in a grid.
    """
    # Validate inputs
    if not isinstance(grids, np.ndarray):
        raise TypeError("Input 'grids' must be a numpy array.")
    if grids.ndim != 3:
        raise ValueError("Input 'grids' must be a 3D array of shape (n, m, m).")
    if grids.shape[1] != grids.shape[2]:
        raise ValueError("Each grid in 'grids' must be square (m x m).")

    # Extract dimensions
    n = grids.shape[0]  # Number of time steps
    m = grids.shape[1]  # Number of agents

    # Iterate over grids and agents to construct paths
    paths = np.zeros((n, 2, m), dtype=float)
    for t in range(n):
        for i in range(m):
            (x, y) = np.where(grids[t, :, :] == i)
            if len(x) == 0 or len(y) == 0:
                raise ValueError(
                    f"Agent {i} is missing in grid at time step {t}. Each agent must "
                    "be present in each grid time step."
                )
            elif len(x) > 1 or len(y) > 1:
                raise ValueError(
                    f"Agent {i} appears multiple times in grid at time step {t}. Each "
                    "agent must appear at most once in each grid time step."
                )
            paths[t, 0, i] = x[0]  # x-coordinate
            paths[t, 1, i] = y[0]  # y-coordinate

    return paths


def paths2grids(paths: np.ndarray) -> np.ndarray:
    """
    Convert a list of paths to a list of grids.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.

    Returns:
        np.ndarray: A 3D array of shape (n, m, m) representing n time steps of m x m
            grids. Each grid represents the relative positions of m agents at a given
            time step. Each row and column of the grid can contain at most one non-zero
            entry corresponding to an index i.

    Raises:
        TypeError: If the input 'paths' is not a numpy array.
        ValueError: If the input 'paths' does not have the correct shape.
        ValueError: If any agent is missing or appears multiple times in a grid.
    """
    # Validate inputs
    if not isinstance(paths, np.ndarray):
        raise TypeError("Input 'paths' must be a numpy array.")
    if paths.ndim != 3:
        raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
    if paths.shape[1] != 2:
        raise ValueError(
            "The second dimension of 'paths' must be 2 for (x, y) coordinates."
        )

    # Extract dimensions
    n = paths.shape[0]  # Number of time steps
    m = paths.shape[2]  # Number of agents

    # Initialize grids
    grids = np.zeros((n, m, m), dtype=float)

    # Iterate over paths and agents to construct grids
    for t in range(n):
        sorted_x = np.argsort(paths[t, 0, :])
        sorted_y = np.argsort(paths[t, 1, :])
        for i in range(m):
            x_rank = np.where(sorted_x == i)[0][0]
            y_rank = np.where(sorted_y == i)[0][0]
            grids[t, x_rank, y_rank] = i

    # Consistency check: Ensure each agent appears exactly once in each grid time step
    for t in range(n):
        for i in range(m):
            if np.sum(grids[t, :, :] == i) != 1:
                raise ValueError(
                    f"Agent {i} does not appear exactly once in grid at time step {t}."
                )

    # Return grid sequence
    return grids


def paths2windings(
    paths: np.ndarray,
    upscale_factor: int = 1,
    intermediate_shape: str = "linear",
) -> np.ndarray:
    """
    Convert a list of paths to a list of winding numbers.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.
        upscale_factor (int, optional): A factor to upscale the paths by adding more
            intermediate points between time steps. At these points the winding numbers
            are not integers and are not topological invariants, but they can be
            used during the motion planning to guide the agents between timesteps.
            Default is 1 (no upscaling).
        intermediate_shape (str, optional): The shape of the curve to use between
            time steps when upscaling. Can be 'linear' for straight lines or 'spline'
            for cubic splines. Default is 'linear'.

    Returns:
        np.ndarray: A 3D array of shape (n, m, m) representing n time steps of m x m
            winding number grids. Each entry windings[t, i, j] gives the winding number
            of agent i with respect to agent j at time step t. Each m x m matrix is
            skew-symmetric.

    Raises:
        TypeError: If the input 'paths' is not a numpy array.
        ValueError: If the input 'paths' does not have the correct shape.
        ValueError: If 'upscale_factor' is not a positive integer.
        ValueError: If 'intermediate_shape' is not 'linear' or 'spline'.

    Note:
        The winding numbers are multiplied by 2 to make so that they are integers at
        the end of each timestep (i.e., when they are topological invariants for the
        braids corresponding to the input paths).
    """
    # Validate inputs
    if not isinstance(paths, np.ndarray):
        raise TypeError("Input 'paths' must be a numpy array.")
    if paths.ndim != 3:
        raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
    if paths.shape[1] != 2:
        raise ValueError(
            "The second dimension of 'paths' must be 2 for (x, y) coordinates."
        )
    if not isinstance(upscale_factor, int) or upscale_factor < 1:
        raise ValueError("Input 'upscale_factor' must be a positive integer.")
    if intermediate_shape not in ["linear", "spline"]:
        raise ValueError(
            "Input 'intermediate_shape' must be either 'linear' or 'spline'."
        )

    # Extract dimensions
    n = paths.shape[0]  # Number of time steps
    m = paths.shape[2]  # Number of agents

    # Initialize variables
    windings: np.ndarray = np.zeros((n, m, m), dtype=float)
    theta: np.ndarray = np.zeros((m, m), dtype=float)  # N×N matrix of angles at time k
    theta_prev: np.ndarray = np.zeros((m, m), dtype=float)  # angles at time step k-1
    delta_theta: np.ndarray = np.zeros((m, m), dtype=float)  # matrix of angle diff

    # Iterate over time steps to compute winding numbers
    for t in range(1, n):

        # Compute the angles variation with respect to the previous time step
        dx = paths[t, 0, :][np.newaxis, :] - paths[t, 0, :][:, np.newaxis]  # N×N
        dy = paths[t, 1, :][np.newaxis, :] - paths[t, 1, :][:, np.newaxis]  # N×N
        theta = np.arctan2(dy, dx)
        delta_theta = theta - theta_prev

        # Compute and store winding numbers for current time step
        windings[t, :, :] = windings[t - 1, :, :] + delta_theta

        # Update previous angles
        theta_prev = theta

    return windings
