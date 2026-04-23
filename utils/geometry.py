import numpy as np


def check_positions(
    positions: np.ndarray,
    d_min: float = 0.1,
    dynamics: str = "single_integrator",
    u_max: np.ndarray = None,
    dt: float = 0.1,
    verbose: bool = False,
) -> bool:
    """
    Check that the given positions are admissible for collision avoidance.

    Args:
        positions: A 2-by-m or 3-by-m array of initial or goal positions of m agents.
        d_min: The minimum distance required between any two agents.
        dynamics: The type of dynamics of the agents {"single_integrator", "unicycle"}.
        u_max: The maximum control input for each agent, as a 2-by-m array.
        dt: The time step for the simulation.
        verbose: If True, print verbose information.
    """
    # Inflate d_min depending on the dynamics
    d: float
    if dynamics == "single_integrator":
        if u_max is not None:
            v_max = np.sqrt(u_max[0, 0] ** 2 + u_max[0, 1] ** 2)
            d = np.sqrt(d_min**2 + (v_max * dt) ** 2)
        else:
            d = d_min
    elif dynamics == "unicycle":
        if u_max is not None:
            d = d_min + u_max[0, 0] * dt
        else:
            d = d_min

    # Check pairwise distances
    distances_ok: bool = True
    m = positions.shape[1]
    for i in range(m):
        for j in range(i + 1, m):
            dist = np.linalg.norm(positions[:, i] - positions[:, j])
            if dist < d:
                distances_ok = False
                if verbose:
                    print(
                        f"Agents {i} and {j} are too close: "
                        f"distance = {dist:.4f} < d = {d:.4f}"
                    )

    return distances_ok
