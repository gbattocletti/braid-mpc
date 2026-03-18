"""
Visualization utilities for plotting paths, grids, and winding numbers.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_paths(paths: np.ndarray) -> None:
    """
    Plot a list of paths.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.

    Returns:
        None
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
    m = paths.shape[2]  # Number of agents

    # Create the plot
    _, ax = plt.subplots(figsize=(10, 10))
    for i in range(m):
        x = paths[:, 0, i]
        y = paths[:, 1, i]
        ax.plot(x, y, linewidth=1.3, label=f"Agent {i}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("t")
    ax.legend()
    ax.grid(True)
    plt.show()

    # TODO: return fig and ax?
