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
    u_prev: np.ndarray | None = None,
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
        v_max (np.ndarray): (2, ) max per-step displacement along x, y
        a_max (np.ndarray): (2, ) max change in per-step displacement along x, y

    Returns:
        x (np.ndarray): (2, K+1) optimal trajectory from x_init to x_goal
        u (np.ndarray): (2, K) optimal displacements
    """
    # TODO include u_prev
    if u_prev is not None:
        pass

    # Initialize ocp
    ocp = ca.Opti()
    ocp.solver(solver, ocp_options, solver_options)
    x = ocp.variable(2, K + 1)

    # Boundary conditions
    ocp.subject_to(x[:, 0] == x_init)

    # State and input constraints
    u = x[:, 1 : K + 1] - x[:, 0:K]  # velocity (input of single integrator)
    uu = u[:, 1:K] - u[:, 0 : K - 1]  # acceleration (change in velocity)
    ocp.subject_to(ocp.bounded(-u_max[0] * dt, u[0, :], u_max[0] * dt))
    ocp.subject_to(ocp.bounded(-u_max[1] * dt, u[1, :], u_max[1] * dt))
    ocp.subject_to(ocp.bounded(-u_rate_max[0] * dt, uu[0, :], u_rate_max[0] * dt))
    ocp.subject_to(ocp.bounded(-u_rate_max[1] * dt, uu[1, :], u_rate_max[1] * dt))

    # Objective (quadratic cost)
    cost_function = 0
    for k in range(K):
        cost_function += u[:, k].T @ np.diag([1, 1]) @ u[:, k]  # control effort
        cost_function += ca.sumsqr(x[:, k + 1] - x_goal)  # distance to goal
    ocp.minimize(cost_function)

    # Solve the optimization problem and return the optimal trajectory and displacements
    sol = ocp.solve()
    return (
        sol.value(x),
        sol.value(u),
        sol.value(cost_function),
        sol.stats()["t_proc_total"],
    )
