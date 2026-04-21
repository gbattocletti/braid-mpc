"""
Visualisation utilities for debugging the DistributedMPC controller.
"""

import matplotlib.pyplot as plt
import numpy as np

from core.mpc_distributed import DistributedMPC


def half_plane(
    p_i: np.ndarray, p_j: np.ndarray, d: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (a_ij, b_ij, d/2) s.t. the ego's feasible set is a_ij^T @ x_i >=b_ij +  d/2.

    Args:
        p_i: position of the ego agent (2, )
        p_j: position of the other agent (2, )
        d: inflated safety distance to maintain between agents

    Returns:
        a: normal vector of the half-plane boundary (2, )
        b: offset of the half-plane boundary (scalar)
        d: margin to maintain from half-plane boundary, equal to half input d (scalar)

    Raises:
        ValueError: if p_i and p_j are too close to compute a meaningful half-plane.
    """
    # Preprocess positions
    diff = p_i - p_j
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-9:
        raise ValueError("Agents are too close to compute a meaningful half-plane.")

    # Compute half-plane parameters
    a_ij = diff / diff_norm
    b_ij = a_ij @ (p_i + p_j).T / 2 + d / 2

    return a_ij, b_ij


def draw_half_plane(
    ax: plt.Axes,
    a_ij: np.ndarray,
    b_ij: float,
    xlim: np.ndarray,
    ylim: np.ndarray,
    color: str = "#444444",
    alpha_line: float = 0.7,
    alpha_fill: float = 0.025,
    show_midline: bool = False,
    d_half: float = 0.0,
) -> None:
    """
    Draw the hyperplane a_ij^T @ x >= b_ij.

    The function plots:
        a_ij^T @ x == b_ij as a solid line
        a_ij^T @ x < b_ij as a shaded region

    Args:
        ax (matplotlib.axes.Axes): the matplotlib axes to draw on.
        a_ij (np.ndarray): normal vector of the half-plane boundary (2, )
        b_ij (float): offset of the half-plane boundary (scalar)
        xlim (np.ndarray): (x_min, x_max) for plotting limits
        ylim (np.ndarray): (y_min, y_max) for plotting limits
        color (str, optional): color for hyperplane and shaded area
        alpha_line (float, optional): alpha value for the hyperplane line
        alpha_fill (float, optional): alpha value for the shaded area
        show_midline (bool, optional): whether to show the midline without d_half margin
        d_half (float, optional): safety margin to maintain from half-plane boundary

    Returns:
        None
    """
    # Preprocess inputs
    tang = np.array([-a_ij[1], a_ij[0]])
    centre = np.array([0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1])])
    a_centre = float(a_ij @ centre)
    diag = float(np.hypot(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    span = abs(b_ij - a_centre) + 2.0 * diag

    # Boundary a^T x == b
    anchor = centre + (b_ij - a_centre) * a_ij
    x_boundary_1 = anchor - span * tang
    x_boundary_2 = anchor + span * tang
    ax.plot(
        [x_boundary_1[0], x_boundary_2[0]],
        [x_boundary_1[1], x_boundary_2[1]],
        color=color,
        linewidth=1,
        alpha=alpha_line,
        zorder=3,
    )

    # Forbidden region a^T x < b
    poly_verts = np.array(
        [
            x_boundary_1,
            x_boundary_2,
            x_boundary_2 - span * a_ij,
            x_boundary_1 - span * a_ij,
        ]
    )
    ax.fill(
        poly_verts[:, 0],
        poly_verts[:, 1],
        color=color,
        alpha=alpha_fill,
        linewidth=0,
        zorder=1,
    )

    # midline  a^T x = b_ij - d_half
    if show_midline and d_half > 0.0:
        anchor_mid = centre + (b_ij - d_half - a_centre) * a_ij
        x_midline_1 = anchor_mid - span * tang
        x_midline_2 = anchor_mid + span * tang
        ax.plot(
            [x_midline_1[0], x_midline_2[0]],
            [x_midline_1[1], x_midline_2[1]],
            color=color,
            lw=0.7,
            ls="--",
            alpha=alpha_line * 0.7,
            zorder=2,
        )


def plot_hyperplanes(mpc: DistributedMPC, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualise the convex collision-avoidance hyperplanes of a DistributedMPC.

    Args:
        mpc: The DistributedMPC instance for which to plot the hyperplanes. Must have
            already been initialized with initialize_ocp().
        figsize (np.ndarray, optional): the size of the figure to create
            (default: [10, 10]).
        show_solution (bool, optional): whether to show the solution trajectory
            (default: False).
        show_legend (bool, optional): whether to show legend (default: False).
        show (bool, optional): whether to call plt.show() (default: False).
        block (bool, optional): whether to block execution when calling plt.show()
            (default: False).

    Returns:
        fig: The matplotlib figure containing the plot.
        ax: The matplotlib axes containing the plot.

    Raises:
        ValueError: If mpc is not a DistributedMPC instance or if mpc.ocp has not been
            solved yet.
    """
    # Parse kwargs
    figsize: np.ndarray = kwargs.get("figsize", np.array([10, 10]))
    show_solution: bool = kwargs.get("show_solution", False)
    show_legend: bool = kwargs.get("show_legend", False)
    show: bool = kwargs.get("show", False)
    block: bool = kwargs.get("block", False)

    # Check mpc type and status
    if mpc.architecture != "distributed":
        raise ValueError(
            "plot_hyperplanes only supports DistributedMPC; got architecture "
            f"'{mpc.architecture}'."
        )
    if mpc.ocp_ready is False:
        raise RuntimeError("MPC.ocp is not initialized. Call initialize_ocp() first.")

    # Extract info from mpc
    m = mpc.m
    K = mpc.K
    x_prev = np.asarray(mpc.ocp.value(mpc.x_prev))[:, :2]  # (K+1, 2)
    x_pred = [np.asarray(mpc.ocp.value(p))[:, :2] for p in mpc.x_pred]  # m-1 x (K+1, 2)
    x_ego = np.asarray(mpc.ocp.value(mpc.x_0))[:2]  # (2, )
    x_others = np.stack([p[0, :2] for p in x_pred], axis=-1)  # (2, m-1)

    # Compute inflated safety distance depending on the dynamics and control limits
    d: float
    if mpc.d_min is None:
        raise ValueError("mpc.d_min must be set.")
    if mpc.u_max is None:
        d = mpc.d_min
    else:
        u_max = np.asarray(mpc.u_max).reshape(-1)
        if mpc.dynamics == "single_integrator":
            v_max = float(np.hypot(u_max[0], u_max[1]))
            d = float(np.sqrt(mpc.d_min**2 + (v_max * mpc.dt) ** 2))
        elif mpc.dynamics == "unicycle":
            d = float(mpc.d_min + u_max[0] * mpc.dt)
        else:
            raise ValueError(f"Unknown dynamics: {mpc.dynamics!r}.")

    # Compute x and y limits for plotting
    if mpc.x_min is not None and mpc.x_max is not None:
        x_min = np.asarray(mpc.x_min).reshape(-1)
        x_max = np.asarray(mpc.x_max).reshape(-1)
        xlim = (float(x_min[0]), float(x_max[0]))
        ylim = (float(x_min[1]), float(x_max[1]))
    else:
        all_pos = np.concatenate([x_ego, x_others.reshape(-1, 2)], axis=0)
        pad = 1.0
        xlim = (float(all_pos[:, 0].min() - pad), float(all_pos[:, 0].max() + pad))
        ylim = (float(all_pos[:, 1].min() - pad), float(all_pos[:, 1].max() + pad))

    # Define colormap
    colors = plt.color_sequences["tab10"][:m]  # color 0 for ego, 1: for others

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=figsize / 2.54)

    # Ego agent current location
    # Plot current location
    ax.plot(
        x_prev[0, 0],
        x_prev[0, 1],
        color=colors[0],
        alpha=1,
        marker="D",
        markersize=6,
        zorder=6,
    )

    # Show ego agent solution at previous time step (used to compute the hyperplanes)
    ax.plot(
        x_prev[:, 0],
        x_prev[:, 1],
        color=colors[0],
        lw=1.5,
        alpha=1,
        label="Agent 0 sol k-1",
        marker="D",
        markersize=4,
        zorder=5,
    )

    # Show new solution
    if show_solution is True and mpc.sol is not None:
        x_sol = np.asarray(mpc.sol.value(mpc.x))[:, :2]
        ax.plot(
            x_sol[:, 0],
            x_sol[:, 1],
            color=colors[0],
            lw=1.5,
            alpha=1,
            label="Agent 0 sol k",
            marker="o",
            markersize=4,
            zorder=4,
        )

    # Plot other agents' hyperplanes, current location and trajectories
    for j in range(m - 1):
        color = colors[j + 1]  # color of agent j
        # Plot hyperplanes for agent j
        for k in range(K + 1):
            a_ij, b_ij = half_plane(x_prev[k, :], x_pred[j][k, :], d)
            draw_half_plane(ax, a_ij, b_ij, xlim, ylim, color=color)

        # Plot current location
        ax.plot(
            x_pred[j][0, 0],
            x_pred[j][0, 1],
            color=color,
            alpha=1,
            marker="D",
            markersize=6,
            zorder=6,
        )

        #  Plot trajectory
        ax.plot(
            x_pred[j][:, 0],
            x_pred[j][:, 1],
            color=color,
            lw=1.5,
            alpha=1,
            label=f"Agent {j+2}",
            marker="D",
            markersize=4,
            zorder=5,
        )

    # Set axes limits, labels and grid
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="major", linestyle=":", color="gray", linewidth=0.5, zorder=0)
    ax.grid(True, which="minor", linestyle=":", color="gray", linewidth=0.3, zorder=0)

    # Show legend
    if show_legend is True:
        ax.legend(loc="upper right")

    # Show plot
    if show:
        plt.show(block=block)

    return fig, ax
