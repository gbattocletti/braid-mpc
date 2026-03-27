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
        self.x = self.ocp.variable(self.K + 1, self.n_x, self.m)
        self.u = self.ocp.variable(self.K, self.n_u, self.m)

        # Initialize OCP parameters
        self.x_0 = self.ocp.parameter(self.n_x, self.m)
        self.x_goal = self.ocp.parameter(self.n_x, self.m)
        self.w_curr = self.ocp.parameter(self.m, self.m)
        self.w_target = self.ocp.parameter(self.m, self.m)

        # Constraints
        # Initial state constraint
        self.ocp.subject_to(self.x[0, :, :] == self.x_0)

        # Dynamics
        for k in range(self.K):
            for i in range(self.m):
                x_next = self.x[k, :, i] + self.dxdt(self.u[k, :, i]) * self.dt
                self.ocp.subject_to(self.x[k + 1, :, i] == x_next)

        # State constraints
        if self.x_min is not None and self.x_max is not None:
            for k in range(self.K + 1):
                for i in range(self.m):
                    self.ocp.subject_to(self.x_min <= self.x[k, :, i])
                    self.ocp.subject_to(self.x[k, :, i] <= self.x_max)

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(self.u_min <= self.u[k, :, i])
                    self.ocp.subject_to(self.u[k, :, i] <= self.u_max)

        # Input rate constraints
        # NOTE: the previous u (i.e., the one applied at the previous time step) should
        # also be taken into account to constraint u[0], but for simplicity we only
        # constraint the rate between the optimization variables (at least for now).
        if self.u_rate_min is not None and self.u_rate_max is not None:
            for i in range(self.m):
                for k in range(self.K - 1):
                    self.ocp.subject_to(
                        self.u_rate_min <= self.u[k + 1, :, i] - self.u[k, :, i]
                    )
                    self.ocp.subject_to(
                        self.u[k + 1, :, i] - self.u[k, :, i] <= self.u_rate_max
                    )

        # Total input constraints
        if self.u_tot_max is not None:
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(ca.norm_1(self.u[k, :, i]) <= self.u_tot_max)

        # Collision avoidance constraints
        if self.d_min is not None:
            for i in range(self.m):
                for j in range(self.m):
                    if i == j:
                        continue  # skip self-collision
                    for k in range(self.K + 1):
                        dist = ca.norm_2(self.x[k, :, i] - self.x[k, :, j])
                        self.ocp.subject_to(dist >= self.d_min)

        # Cost function
        self.cost_function = 0

        # Goal tracking cost
        if self.alpha_g is not None and self.alpha_g > 0:
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.cost_function += self.alpha_g * (
                        (self.x[k, 0, i] - self.x_goal[0, i]) ** 2
                        + (self.x[k, 1, i] - self.x_goal[1, i]) ** 2
                    )

        # Terminal goal tracking cost
        if self.alpha_g_terminal is not None and self.alpha_g_terminal > 0:
            for i in range(self.m):
                delta_goal: float = (
                    (self.x[self.K, 0, i] - self.x_goal[0, i]) ** 2
                    + (self.x[self.K, 1, i] - self.x_goal[1, i]) ** 2
                ) - (
                    (self.x[0, 0, i] - self.x_goal[0, i]) ** 2
                    + (self.x[0, 1, i] - self.x_goal[1, i]) ** 2
                )
                self.cost_function += self.alpha_g_terminal * delta_goal

        # Control input cost
        if self.alpha_u is not None and self.alpha_u > 0:
            for i in range(self.m):
                for k in range(self.K):
                    self.cost_function += self.alpha_u * (
                        self.u[k, :, i] @ self.R @ self.u[k, :, i].T
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
                            self.x[k, 1, i] - self.x[k, 1, j],
                            self.x[k, 0, i] - self.x[k, 0, j],
                        )
                        theta_prev: ca.SX | ca.MX = ca.atan2(
                            self.x[k - 1, 1, i] - self.x[k - 1, 1, j],
                            self.x[k - 1, 0, i] - self.x[k - 1, 0, j],
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
        x_pred: list[np.ndarray],
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol_prev: ca.OptiSol | None = None,
        **kwargs,
    ) -> ca.OptiSol:
        """
        Solve the NMPC problem.

        Args:
            x_0 (np.ndarray): initial state of the ego agent (n_x, )
            x_goal (np.ndarray): goal state of the ego agent (n_x, )
            x_pred (list[np.ndarray]): predicted states of agents (m-1, n_x, K+1)
            w_curr (np.ndarray): current winding number w.r.t. agents (m-1, )
            w_target (np.ndarray): target winding number w.r.t. agents (m-1, )
            use_warm_start (bool, optional): whether to use the previous solution for
                warm starting
            sol_prev (ca.OptiSol | None, optional): previous solution for warm starting
            **kwargs: Additional keyword arguments
        Returns:
            sol (ca.OptiSol): solution of the OCP

        Raises:
            RuntimeError: if the OCP has not been initialized yet.
            ValueError: if the input arguments have incorrect shapes.
            ValueError: if use_warm_start is True but sol is not provided.
            RuntimeError: if the OCP solver fails to find a solution.
        """
        # TODO
        pass

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
        if self.sol is None:
            raise RuntimeError(
                "No solution available. Please solve the OCP before calling "
                "_check_cost()."
            )

        raise NotImplementedError(
            "The _check_cost() method is not implemented yet for the centralized MPC."
        )
