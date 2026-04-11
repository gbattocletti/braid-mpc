"""
Utility functions to convert between different representations of topological
specifications, such as braids, paths, sequences of grids, and winding numbers.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def angle_diff(
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the difference between two angles alpha and beta, taking into account
    the periodicity of angles.

    Args:
        alpha (float or np.ndarray): The first angle(s) in radians.
        beta (float or np.ndarray): The second angle(s) in radians.

    Returns:
        float or np.ndarray: The difference between alpha and beta, normalized to the
            range (-pi, pi].
    """
    return np.arctan2(np.sin(alpha - beta), np.cos(alpha - beta))


def relative_headings(poses: np.ndarray) -> np.ndarray:
    """
    Compute the relative headings between agents based on their positions.

    Args:
        poses (np.ndarray): A 2D array of shape (3, m) representing the current
            positions of m agents in 2D space. Each entry poses[:, i] gives the
            (x, y, theta) coordinates of agent i.

    Returns:
        np.ndarray: A 2D array of shape (m, m) representing the relative headings
            between each pair of agents. The entry at [i, j] gives the angle from
            agent i to agent j in radians. The relative headings matrix is symmetrical
            with zeros on the main diagonal.
    """
    # Remove heading from poses
    poses = poses[:2, :]  # (2, m)

    # Compute relative headings
    diff = poses[:, np.newaxis, :] - poses[np.newaxis, :, :]
    headings = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (m, m)

    return headings


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
            (x, y) = np.where(grids[t, :, :] == i + 1)
            if len(x) == 0 or len(y) == 0:
                raise ValueError(
                    f"Agent {i+1} is missing in grid at time step {t}. Each agent must "
                    "be present in each grid time step."
                )
            elif len(x) > 1 or len(y) > 1:
                raise ValueError(
                    f"Agent {i+1} appears multiple times in grid at time step {t}. "
                    "Each agent must appear at most once in each grid time step."
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
            grids[t, x_rank, y_rank] = i + 1

    # Consistency check: Ensure each agent appears exactly once in each grid time step
    for t in range(n):
        for i in range(m):
            if np.sum(grids[t, :, :] == i + 1) != 1:
                raise ValueError(
                    f"Agent {i+1} does not appear exactly once in "
                    f"grid at time step {t}."
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

    Notes:
        - If using the 'spline' interpolation method for upscaling, the resuling paths
          may not correspond to the same braid as the original paths, since the spline
          interpolation can introduce additional crossings between agents.
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

    # Upscale paths if needed
    if upscale_factor > 1:
        n, _, m = paths.shape
        new_n = (n - 1) * upscale_factor + 1
        upscaled_paths = np.zeros((new_n, 2, m), dtype=float)
        for i in range(m):
            if intermediate_shape == "linear":
                upscaled_paths[:, 0, i] = np.interp(
                    np.linspace(0, n - 1, new_n),
                    np.arange(n),
                    paths[:, 0, i].T,
                ).T
                upscaled_paths[:, 1, i] = np.interp(
                    np.linspace(0, n - 1, new_n),
                    np.arange(n),
                    paths[:, 1, i].T,
                ).T
            elif intermediate_shape == "spline":
                cs_x = CubicSpline(np.arange(n), paths[:, 0, i])
                cs_y = CubicSpline(np.arange(n), paths[:, 1, i])
                upscaled_paths[:, 0, i] = cs_x(np.linspace(0, n - 1, new_n))
                upscaled_paths[:, 1, i] = cs_y(np.linspace(0, n - 1, new_n))
        paths = upscaled_paths

    # Extract dimensions
    n, _, m = paths.shape

    # Initialize variables
    windings: np.ndarray = np.zeros((n, m, m), dtype=float)
    theta: np.ndarray = np.zeros((m, m), dtype=float)  # N×N matrix of angles at time k
    theta_prev: np.ndarray = np.zeros((m, m), dtype=float)  # angles at time step k-1
    delta_theta: np.ndarray = np.zeros((m, m), dtype=float)  # matrix of angle diff

    # Iterate over time steps to compute winding numbers
    for t in range(0, n):

        # Compute the current angles between the robots
        dx = paths[t, 0, :][np.newaxis, :] - paths[t, 0, :][:, np.newaxis]  # N×N
        dy = paths[t, 1, :][np.newaxis, :] - paths[t, 1, :][:, np.newaxis]  # N×N
        theta = np.arctan2(dy, dx)  # N×N matrix of angles between robots at time k
        theta = np.nan_to_num(theta)  # replace NaNs with 0 (when dx=dy=0)

        # Skip computation of winding numbers for t=0 as previous theta is undefined
        if t > 0:
            # Compute the angles variation with respect to the previous time step
            delta_theta = 1 / (2 * np.pi) * angle_diff(theta, theta_prev)
            windings[t, :, :] = windings[t - 1, :, :] + delta_theta

        # Update previous angles
        theta_prev = theta

    return windings


def compute_winding_weights(
    positions: np.ndarray[float],
    d_threshold: float | None = None,
) -> np.ndarray:
    """
    Compute weights for the winding number cost term based on the distances
    between agents. The weights are higher when agents are closer together and
    lower when they are farther apart.

    Args:
        positions (np.ndarray): A 2D array of shape (m, 2) representing the current
            positions of m agents in 2D space. Each entry positions[i, :] gives the
            (x, y) coordinates of agent i.
        d_threshold (float, optional): Distance after which the cost is considered
            0. If None, the cost is not capped. Default is None.

    Returns:
        np.ndarray: A 2D array of shape (m, m) representing the weights for the
            winding number cost between each pair of agents. The weights matrix is
            symmetric, and has zeros on the main diagonal.
    """
    # Extract number of agents
    m = positions.shape[0]

    # Remove heading from positions
    positions = positions[:, :2]  # (m, 2)

    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)  # (m, m)

    # Compute weights
    weights: np.ndarray[float] = np.zeros((m, m), dtype=float)
    weights[distances != 0] = 1 / (distances[distances != 0] ** 2)
    if d_threshold is not None:
        weights[distances > d_threshold] = 0.0

    # Return weights
    return weights
