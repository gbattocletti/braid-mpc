"""
Visualization utilities for plotting paths, grids, and winding numbers.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_paths_2d(paths: np.ndarray, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of paths in 2D space.

    Args:
        paths (np.ndarray): A 3D array of shape (n, 2, m) representing n time steps of
            m agents' positions in 2D space. Each entry paths[t, :, i] gives the
            (x, y) coordinates of agent i at time step t.
        show (bool, optional): Whether to display the plot. Default is False.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot.
    """
    # Parse kwargs
    show = kwargs.get("show", False)
    figsize = kwargs.get("figsize", np.array([10, 10]))

    # Validate inputs
    if not isinstance(paths, np.ndarray):
        raise TypeError("Input 'paths' must be a numpy array.")
    if paths.ndim != 3:
        raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
    if paths.shape[1] != 2:
        raise ValueError(
            "The second dimension of 'paths' must be 2 for (x, y) coordinates."
        )
    if not isinstance(figsize, np.ndarray):
        if not isinstance(figsize, (tuple, list)):
            raise TypeError("Input 'figsize' must be a numpy array, tuple, or list.")
        figsize = np.array(figsize)

    # Extract dimensions
    n, _, m = paths.shape

    # Create the plot
    colors = plt.color_sequences["tab10"][:m]
    fig, ax = plt.subplots(figsize=figsize / 2.54)
    for t in range(n):
        for i in range(m):
            x = paths[t, 0, i]
            y = paths[t, 1, i]
            ax.plot(x, y, linewidth=1.3, color=colors[i], label=f"Agent {i+1}")

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

    if show is True:
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
        show (bool, optional): Whether to display the plot. Default is False.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot

    Raises:
        TypeError: If the input 'paths' is not a numpy array.
        ValueError: If the input 'paths' does not have the correct shape.
        ValueError: If the second dimension of 'paths' is not 2 for (x, y) coordinates.
    """
    # Parse kwargs
    figsize = kwargs.get("figsize", np.array([10, 10]))
    pov = kwargs.get("pov", [15, 35, 0])
    show = kwargs.get("show", False)

    # Validate inputs
    if not isinstance(paths, np.ndarray):
        raise TypeError("Input 'paths' must be a numpy array.")
    if paths.ndim != 3:
        raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
    if paths.shape[1] != 2:
        raise ValueError(
            "The second dimension of 'paths' must be 2 for (x, y) coordinates."
        )
    if not isinstance(figsize, np.ndarray):
        if not isinstance(figsize, (tuple, list)):
            raise TypeError("Input 'figsize' must be a numpy array, tuple, or list.")
        figsize = np.array(figsize)

    # Preprocess
    n, _, m = paths.shape  # Extract dimensions
    time = np.arange(n)  # Create time array for z-axis
    colors = plt.color_sequences["tab10"][:m]  # Define colormaps

    # Normalize space to mxm square for consistent visualization
    paths_max = np.max(np.max(paths, axis=0), axis=1)
    paths[:, 0, :] = paths[:, 0, :] / paths_max[0] * m
    paths[:, 1, :] = paths[:, 1, :] / paths_max[1] * m

    # Create the plot
    fig: plt.Figure = plt.figure(figsize=figsize / 2.54)
    ax: Axes3D = fig.add_subplot(projection="3d")
    for i in range(m):
        x = paths[:, 0, i]
        y = paths[:, 1, i]
        ax.plot(x, y, time, linewidth=1.3, color=colors[i], label=f"Agent {i+1}")

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

    if show:
        plt.show()

    return fig, ax


def plot_windings(windings: np.ndarray, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the winding numbers of a braid over time.

    Args:
        windings (np.ndarray): A 3D array of shape (n, m, m) representing n time steps
            of m agents' winding numbers. Each entry windings[t, i, j] gives the winding
            number of agent i with respect to agent j at time step t.
        figsize (tuple[float, float], optional): The size of the figure in cm.
        show_legends (bool, optional): Whether to show legends for the plot. Default is
            False.
        show (bool, optional): Whether to display the plot. Default is False.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects for the plot.

    Raises:
        TypeError: If the input 'windings' is not a numpy array.
        ValueError: If the input 'windings' does not have the correct shape.
    """
    # Parse kwargs
    figsize = kwargs.get("figsize", np.array([10, 10]))
    show_legends = kwargs.get("show_legends", False)
    show = kwargs.get("show", False)

    # Validate inputs
    if not isinstance(windings, np.ndarray):
        raise TypeError("Input 'windings' must be a numpy array.")
    if windings.ndim != 3 or windings.shape[1] != windings.shape[2]:
        raise ValueError("Input 'windings' must be a 3D array of shape (n, m, m).")
    if not isinstance(figsize, np.ndarray):
        if not isinstance(figsize, (tuple, list)):
            raise TypeError("Input 'figsize' must be a numpy array, tuple, or list.")
        figsize = np.array(figsize)

    # Extract dimensions
    n, m, _ = windings.shape

    # Create plot variables
    time = np.arange(n)  # Create time array for z-axis
    colors = plt.color_sequences["tab10"][:m]  # Define colormaps

    # Create the plot
    plot_cols = 3
    plot_rows = m // 3 + (m % 3 > 0)
    n_plots = plot_rows * plot_cols
    fig: plt.Figure
    axes: np.ndarray[plt.Axes]
    fig, axes = plt.subplots(
        nrows=plot_rows,
        ncols=plot_cols,
        figsize=figsize / 2.54,
        sharex=True,
    )
    for i in range(n_plots):
        row = i // plot_cols
        col = i % plot_cols
        ax: plt.Axes = axes[row, col]
        if i >= m:
            ax.axis("off")  # Hide unused subplots
            continue
        for j in range(m):
            if j == i:
                continue  # Skip self-winding numbers
            ax.plot(
                time,
                windings[:, i, j],
                linewidth=1.5,
                color=colors[j],
                label=f"w_{{{i}{j}}}",
            )

        # Set labels and title
        ax.set_xlim(0, n - 1)
        ax.set_ylim(windings.min() - 1, windings.max() + 1)
        ax.grid(
            True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=1
        )
        ax.grid(
            True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=1
        )
        ax.minorticks_on()
        ax.set_title(f"Agent {i+1}")
        ax.set_xlabel("t")
        ax.set_ylabel("$w_{ij}$")
        ax.grid(True)

        if show_legends:
            ax.legend()

    if show:
        plt.show()

    return fig, axes
