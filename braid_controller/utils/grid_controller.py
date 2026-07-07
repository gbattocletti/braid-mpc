import casadi as ca
import numpy as np

ocp_options: dict = {
    "expand": True,
    "error_on_fail": False,  # suppress compiler output
    "print_time": 0,  # suppress compiler output
    "verbose": False,  # suppress compiler output
    "record_time": True,  # record time statistics
}
solver: str = "ipopt"
solver_options: dict = {
    "max_iter": 10_000,
    "max_wall_time": 100.0,  # [s]
    "max_cpu_time": 100.0,  # [s]
    "print_level": 0,  # 0-5 (0 = silent, 5 = verbose)
    "tol": 1e-6,  # optimality tolerance
    "acceptable_tol": 1e-4,  # tolerance for early termination
    "linear_solver": "mumps",  # recommended for most problems
}


def grid_controller(
    x_init: np.ndarray,
    x_goal: np.ndarray,
    K: int,
    dt: float,
    u_max: np.ndarray,
    u_rate_max: np.ndarray,
    u_prev: np.ndarray,
):
    """
    Controller using CasADi to solve a quadratic program for smooth straight-line
    motion. Lines are guaranteed to be collision-free so no collision avoidance is
    needed. In order to effectively enforce input constraints, a K>1 prediction
    horizon is used.

    Args:
    positions
        x_init (np.ndarray): (2,) current position
        x_goal (np.ndarray): (2,) target position
        K (int): horizon length (number of steps)
        dt (float): time step duration
        u_max (np.ndarray): (2, ) max per-step displacement along x, y (control input)
        u_rate_max (np.ndarray): (2, ) max change in per-step displacement along x, y
        u_prev (np.ndarray): (2, ) control input (displacement) at previous time step

    Returns:
        x (np.ndarray): (2, K+1) optimal trajectory from x_init to x_goal
        u (np.ndarray): (2, K) optimal displacements
    """
    # Initialize ocp
    ocp = ca.Opti()
    ocp.solver(solver, ocp_options, solver_options)
    x = ocp.variable(2, K + 1)
    u = ocp.variable(2, K)

    # Boundary conditions
    ocp.subject_to(x[:, 0] == x_init)

    # Dynamics
    for k in range(K):
        x_next = x[:, k] + u[:, k] * dt
        ocp.subject_to(x[:, k + 1] == x_next)

    # Input constraints
    for k in range(K):
        ocp.subject_to(-u_max <= u[:, k])
        ocp.subject_to(u[:, k] <= u_max)
    ocp.subject_to(-u_rate_max <= u[:, 0] - u_prev)
    ocp.subject_to(u[:, 0] - u_prev <= u_rate_max)
    for k in range(K - 1):
        ocp.subject_to(-u_rate_max <= u[:, k + 1] - u[:, k])
        ocp.subject_to(u[:, k + 1] - u[:, k] <= u_rate_max)

    # Objective (quadratic cost)
    cost_function = 0
    alpha_u = 1
    alpha_g = 100
    for k in range(K):
        cost_function += alpha_u * u[:, k].T @ np.diag([1, 1]) @ u[:, k]
    for k in range(1, K + 1):
        cost_function += alpha_g * (
            (x[0, k] - x_goal[0]) ** 2 + (x[1, k] - x_goal[1]) ** 2
        )
    ocp.minimize(cost_function)

    # Solve the optimization problem and return the optimal trajectory and displacements
    sol = ocp.solve()
    return (
        sol.value(x),
        sol.value(u),
        sol.value(cost_function),
        sol.stats()["t_proc_total"],
    )
