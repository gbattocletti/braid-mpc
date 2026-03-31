import warnings

import casadi as ca
import numpy as np

from core.mpc import MPC


class CentralizedMPC(MPC):
    """
    Centralized MPC controller class.

    The main structure of the class and of the OCP formulation is the same as for the
    base MPC class, but the OCP is formulated in a centralized way, meaning that the
    cost, dynamics, and constraints, of all the agents are included in a single
    optimization problem. This also means that, differently from the local MPC
    controller, the variables and parameters of this class contain the data for all the
    agents, and thus tehy have an extra dimension of length m(number of agents).
    Additionally, the centralized OCP does not need the predicted trajectories of the
    other agents, as all the trajectories are optimized simultaneously.
    """

    def __init__(self) -> None:
        super().__init__()
        self.solver_options["max_wall_time"] = 600.0  # [s]
        self.solver_options["max_cpu_time"] = 600.0  # [s]

    def _initialize_ocp(self) -> None:
        """
        Initialize the centralized OCP problem for the centralized MPC architecture.
        """
        # Initialize optimization variables
        self.x = [self.ocp.variable(self.K + 1, self.n_x) for _ in range(self.m)]
        self.u = [self.ocp.variable(self.K, self.n_u) for _ in range(self.m)]

        # Initialize OCP parameters
        self.x_0 = self.ocp.parameter(self.n_x, self.m)
        self.x_goal = self.ocp.parameter(self.n_x, self.m)
        self.w_curr = self.ocp.parameter(self.m, self.m)
        self.w_target = self.ocp.parameter(self.m, self.m)

        # Constraints
        # Initial state constraint
        for i in range(self.m):
            self.ocp.subject_to(self.x[i][0, :] == self.x_0[:, i])

        # Dynamics
        for i in range(self.m):
            for k in range(self.K):
                x_next = self.x[i][k, :] + self.dxdt(self.u[i][k, :]) * self.dt
                self.ocp.subject_to(self.x[i][k + 1, :] == x_next)

        # State constraints
        if self.x_min is not None and self.x_max is not None:
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.ocp.subject_to(self.x_min <= self.x[i][k, :])
                    self.ocp.subject_to(self.x[i][k, :] <= self.x_max)

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(self.u_min <= self.u[i][k, :])
                    self.ocp.subject_to(self.u[i][k, :] <= self.u_max)

        # Input rate constraints
        # NOTE: the previous u (i.e., the one applied at the previous time step) should
        # also be taken into account to constraint u[0], but for simplicity we only
        # constraint the rate between the optimization variables (at least for now).
        if self.u_rate_min is not None and self.u_rate_max is not None:
            for i in range(self.m):
                for k in range(self.K - 1):
                    self.ocp.subject_to(
                        self.u_rate_min <= self.u[i][k + 1, :] - self.u[i][k, :]
                    )
                    self.ocp.subject_to(
                        self.u[i][k + 1, :] - self.u[i][k, :] <= self.u_rate_max
                    )

        # Total input constraints
        if self.u_tot_max is not None:
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(ca.norm_1(self.u[i][k, :]) <= self.u_tot_max)

        # Collision avoidance constraints
        if self.d_min is not None:
            for i in range(self.m):
                for j in range(self.m):
                    if i == j:
                        continue  # skip self-collision
                    for k in range(self.K + 1):
                        dist = ca.norm_2(self.x[i][k, :] - self.x[j][k, :])
                        self.ocp.subject_to(dist >= self.d_min)

        # Cost function
        self.cost_function = 0

        # Goal tracking cost
        if self.alpha_g is not None and self.alpha_g > 0:
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.cost_function += self.alpha_g * (
                        (self.x[i][k, 0] - self.x_goal[i, 0]) ** 2
                        + (self.x[i][k, 1] - self.x_goal[i, 1]) ** 2
                    )

        # Terminal goal tracking cost
        if self.alpha_g_progress is not None and self.alpha_g_progress > 0:
            for i in range(self.m):
                delta_goal: float = (
                    (self.x[i][self.K, 0] - self.x_goal[i, 0]) ** 2
                    + (self.x[i][self.K, 1] - self.x_goal[i, 1]) ** 2
                ) - (
                    (self.x[i][0, 0] - self.x_goal[i, 0]) ** 2
                    + (self.x[i][0, 1] - self.x_goal[i, 1]) ** 2
                )
                self.cost_function += self.alpha_g_progress * delta_goal

        # Control input cost
        if self.alpha_u is not None and self.alpha_u > 0:
            for i in range(self.m):
                for k in range(self.K):
                    self.cost_function += self.alpha_u * (
                        self.u[i][k, :] @ self.R @ self.u[i][k, :].T
                    )

        # Winding cost
        if self.alpha_w is not None and np.any(self.alpha_w > 0):
            for i in range(self.m):
                for j in range(self.m):
                    if i == j:
                        continue  # skip self-winding

                    # Get weight for agent i w.r.t. agent j
                    alpha_w_ij: float
                    if isinstance(self.alpha_w, np.ndarray):
                        self.alpha_w: np.ndarray  # to avoid unsubscriptable-object
                        alpha_w_ij = self.alpha_w[i, j]
                    else:
                        alpha_w_ij = self.alpha_w

                    # Compute winding number w.r.t. j at the end of prediction horizon
                    w = self.w_curr[j]
                    for k in range(1, self.K + 1):
                        theta: ca.SX | ca.MX = ca.atan2(
                            self.x[i][k, 1] - self.x[j][k, 1],
                            self.x[i][k, 0] - self.x[j][k, 0],
                        )
                        theta_prev: ca.SX | ca.MX = ca.atan2(
                            self.x[j][k - 1, 1] - self.x[i][k - 1, 1],
                            self.x[j][k - 1, 0] - self.x[i][k - 1, 0],
                        )
                        w += 1 / (2 * np.pi) * self.angle_diff(theta, theta_prev)

                    # Add winding cost to the total cost function
                    self.cost_function += alpha_w_ij * (self.w_target[i][j] - w) ** 2

        # Define the objective
        self.ocp.minimize(self.cost_function)

        # Set the solver options
        self.ocp.solver(self.solver, self.ocp_options, self.solver_options)

        # Set ready flag to true
        self.ocp_ready = True

    def _solve(
        self,
        x_0: np.ndarray,
        x_goal: np.ndarray,
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol_prev: ca.OptiSol | None = None,
        **kwargs,
    ) -> ca.OptiSol:
        """
        Solve the MPC problem for the centralized MPC architecture.
        """
        # Validate inputs
        if x_0.shape != (self.n_x, self.m):
            raise ValueError(
                f"x_0 must have shape ({self.n_x}, {self.m}), " f"but got {x_0.shape}."
            )
        if x_goal.shape != (self.n_x, self.m):
            raise ValueError(
                f"x_goal must have shape ({self.n_x}, {self.m}), "
                f"but got {x_goal.shape}."
            )
        if w_curr.shape != (self.m, self.m):
            raise ValueError(
                f"w_curr must have shape ({self.m}, {self.m}), but got {w_curr.shape}."
            )
        if w_target.shape != (self.m, self.m):
            raise ValueError(
                f"w_target must have shape ({self.m}, {self.m}), "
                f"but got {w_target.shape}."
            )
        if sol_prev is not None and not isinstance(sol_prev, ca.OptiSol):
            raise ValueError(
                "sol_prev must be of type casadi.OptiSol or None, but got "
                f"{type(sol_prev)}."
            )
        if sol_prev is not None and sol_prev is None:
            raise ValueError("use_warm_start is True but sol_prev was not provided.")

        # Parse kwargs
        x_pred = kwargs.get("x_pred", None)
        if x_pred is not None:
            warnings.warn(
                "x_pred is provided but is not used in the centralized MPC "
                "formulation. The x_pred argument will be ignored."
            )

        # Set parameters in OCP
        self.ocp.set_value(self.x_0, x_0)
        self.ocp.set_value(self.x_goal, x_goal)
        self.ocp.set_value(self.w_curr, w_curr)
        self.ocp.set_value(self.w_target, w_target)

        # Warm start
        if use_warm_start is True and sol_prev is not None:
            self.ocp.set_initial(sol_prev.value_variables())
        elif self.sol is None:
            # warm start on 1st solution to avoid numerical errors
            # See: https://github.com/casadi/casadi/discussions/3539
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-"NaN-detected"in-my-optimization%3F  # pylint: disable=line-too-long
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.ocp.set_initial(self.x[i][k, :], x_0[:, i])
        else:
            # No warm start
            # CHECKME: check if this case leads to issues
            pass

        # Solve OCP and return solution object
        sol = self.ocp.solve()
        return sol

    def _check_cost(self) -> tuple[float, float, float, float]:
        """
        Compute the cost function value for the current solution. This method is mainly
        intended for debugging and analysis purposes, as the cost value is already
        computed by the solver and can be extracted from the solution object.

        Note: this method must be called immediately after solving the OCP as the
            solution gets overwritten upon switching to the next agent. To

        Args:
            None

        Returns:
            cost (float): value of the cost function for the current solution
            goal_cost (float): contribution of the goal tracking term to the total cost
            control_cost (float): contribution of control input term to the total cost
            winding_cost (float): contribution of the winding term to the total cost

        Raises:
            RuntimeError: if the OCP has not been solved yet (no solution available).
            ValueError: if the sum of the cost components does not match the total cost.
        """
        # Check if OCP is initialized
        if self.ocp_ready is False or self.sol is None:
            raise RuntimeError(
                "No solution available. Please solve the OCP before calling "
                "_check_cost()."
            )

        raise NotImplementedError(
            "The _check_cost() method is not implemented yet for the centralized MPC."
        )
