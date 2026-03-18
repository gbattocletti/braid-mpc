"""
Visualization utilities for plotting paths, grids, and winding numbers.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_paths_2d(paths: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of paths in 2D space.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot.
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
    n, _, m = paths.shape

    # Create the plot
    colors = plt.cm.get_cmap("tab10", m).colors
    fig, ax = plt.subplots(figsize=(10, 10))
    for t in range(n):
        for i in range(m):
            x = paths[t, 0, i]
            y = paths[t, 1, i]
            ax.plot(x, y, linewidth=1.3, color=colors[i], label=f"Agent {i}")

    ax.set_aspect("equal", "box")
    ax.set_xlim([-1, m])
    ax.set_ylim([-1, m])
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1)
    ax.minorticks_on()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Paths of Agents")
    ax.legend()
    ax.grid(True)
    plt.show()

    return fig, ax


def plot_paths_3d(paths: np.ndarray, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of paths in 3D space.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.
        figsize (tuple[float, float], optional): The size of the figure in cm.
            Default is (10, 10).
        pov (list[float], optional): A list of three floats representing the elevation,
            azimuth, and roll angles for the 3D plot's point of view.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot

    Raises:
        TypeError: If the input 'paths' is not a numpy array.
        ValueError: If the input 'paths' does not have the correct shape.
        ValueError: If the second dimension of 'paths' is not 2 for (x, y) coordinates.
    """
    # Parse kwargs
    figsize = kwargs.get("figsize", (10, 10))
    pov = kwargs.get("pov", [15, 35, 0])

    # Validate inputs
    if not isinstance(paths, np.ndarray):
        raise TypeError("Input 'paths' must be a numpy array.")
    if paths.ndim != 3:
        raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
    if paths.shape[1] != 2:
        raise ValueError(
            "The second dimension of 'paths' must be 2 for (x, y) coordinates."
        )

    # Preprocess
    n, _, m = paths.shape  # Extract dimensions
    time = np.arange(n)  # Create time array for z-axis
    colors = plt.cm.get_cmap("tab10", m).colors  # Define colormaps

    # Create the plot
    figsize = figsize / 2.54  # convert cm to in
    fig: plt.Figure = plt.figure(figsize=figsize)  # convert cm to in
    ax: Axes3D = fig.add_subplot(projection="3d")
    for i in range(m):
        x = paths[:, 0, i]
        y = paths[:, 1, i]
        ax.plot(x, y, time, linewidth=1.3, color=colors[i], label=f"Agent {i}")

    # Set limits and aspect
    ax.set_xlim(-1, m)
    ax.set_ylim(-1, m)
    ax.set_zlim(0, n - 1)
    ax.set_box_aspect([1, 1, 1.7])  # ax.set_aspect("equalxy")
    ax.view_init(elev=pov[0], azim=pov[1], roll=pov[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title("3D Paths of Agents Over Time")
    ax.legend()
    plt.show()

    return fig, ax


def plot_winding_numbers(windings: np.ndarray, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the winding numbers of a braid over time.

    Args:
        windings (np.ndarray): A 3D array of shape (n, m, m) representing n time steps
            of m agents' winding numbers. Each entry windings[t, i, j] gives the winding
            number of agent i with respect to agent j at time step t.
        figsize (tuple[float, float], optional): The size of the figure in cm.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot.

    Raises:
        TypeError: If the input 'windings' is not a numpy array.
        ValueError: If the input 'windings' does not have the correct shape.
    """
    # Parse kwargs
    figsize = kwargs.get("figsize", (10, 10))

    # Validate inputs
    if not isinstance(windings, np.ndarray):
        raise TypeError("Input 'windings' must be a numpy array.")
    if windings.ndim != 3 or windings.shape[1] != windings.shape[2]:
        raise ValueError("Input 'windings' must be a 3D array of shape (n, m, m).")

    # Extract dimensions
    n, m, _ = windings.shape

    # Create the plot
    figsize = figsize / 2.54  # convert cm to in
    fig, ax = plt.subplots(figsize=figsize)
    # TODO

    return fig, ax
