import warnings

import casadi as ca
import numpy as np

from braid_controller.core.mpc import MPC
from braid_controller.utils import invariants


class CentralizedMPC(MPC):
    """
    Centralized MPC controller class.

    The main structure of the class and of the OCP formulation is the same as for the
    distributed MPC, but the OCP is formulated in a centralized way, meaning that the
    cost, dynamics, and constraints, of all the agents are included in a single
    optimization problem. This also means that, differently from the local MPC
    controller, the variables and parameters of this class contain the data for all the
    agents, and thus they have an extra dimension of length m (number of agents).
    Additionally, the centralized OCP does not need the predicted trajectories of the
    other agents, as all the trajectories are optimized simultaneously.
    """

    def __init__(self, dynamics: str = "single_integrator") -> None:
        super().__init__(dynamics=dynamics)
        self.architecture = "centralized"
        self.solver_options["print_level"] = 0
        self.solver_options["max_iter"] = 10_000
        self.solver_options["max_wall_time"] = 60.0  # [s]
        self.solver_options["max_cpu_time"] = 60.0  # [s]
        self.solver_options["mu_strategy"] = "adaptive"
        self.solver_options["warm_start_init_point"] = "yes"

    def set_alpha_w(self, alpha_w: np.ndarray) -> None:
        """
        Set the winding cost weight.

        Args:
            alpha_w (np.ndarray): winding cost weight matrix of shape (m, m).

        Returns:
            None

        Raises:
            ValueError: if alpha_w is a numpy array with incorrect shape
            TypeError: if alpha_w is not a scalar or a numpy array
        """
        if not isinstance(alpha_w, np.ndarray):
            raise TypeError("alpha_w must be a numpy array.")
        if not alpha_w.shape == (self.m, self.m):
            raise ValueError(
                f"alpha_w must have shape ({self.m}, {self.m}), "
                f"but got {alpha_w.shape}."
            )
        self.ocp.set_value(self.alpha_w, alpha_w)

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
        self.w_target = [self.ocp.parameter(self.m, self.m) for _ in range(self.K + 1)]
        self.alpha_g = self.ocp.parameter()  # scalar
        self.alpha_w = self.ocp.parameter(self.m, self.m)

        # Constraints
        # Initial state constraint
        for i in range(self.m):
            self.ocp.subject_to(self.x[i][0, :] == self.x_0[:, i].T)
            self.constraints.append(f"initial conditions i:{i}")

        # Dynamics
        for i in range(self.m):
            for k in range(self.K):
                x_next = (
                    self.x[i][k, :]
                    + self.dxdt(self.x[i][k, :], self.u[i][k, :]) * self.dt
                )
                self.ocp.subject_to(self.x[i][k + 1, :] == x_next)
                self.constraints.append(f"dynamics i:{i} k:{k}")

        # State constraints
        if self.x_min is not None and self.x_max is not None:
            if self.x_min.shape != (1, self.n_x_pos):
                self.x_min = self.x_min.reshape(1, self.n_x_pos)
            if self.x_max.shape != (1, self.n_x_pos):
                self.x_max = self.x_max.reshape(1, self.n_x_pos)
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.ocp.subject_to(self.x_min <= self.x[i][k, : self.n_x_pos])
                    self.constraints.append(f"state constraints min i:{i} k:{k}")
                    self.ocp.subject_to(self.x[i][k, : self.n_x_pos] <= self.x_max)
                    self.constraints.append(f"state constraints max i:{i} k:{k}")

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            if self.u_min.shape != (1, self.n_u):
                self.u_min = self.u_min.reshape(1, self.n_u)
            if self.u_max.shape != (1, self.n_u):
                self.u_max = self.u_max.reshape(1, self.n_u)
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(self.u_min <= self.u[i][k, :])
                    self.constraints.append(f"input constraints min i:{i} k:{k}")
                    self.ocp.subject_to(self.u[i][k, :] <= self.u_max)
                    self.constraints.append(f"input constraints max i:{i} k:{k}")

        # Input rate constraints
        # NOTE: the previous u (i.e., the one applied at the previous time step) should
        # also be taken into account to constraint u[0], but for simplicity we only
        # constraint the rate between the optimization variables (at least for now).
        if self.u_rate_min is not None and self.u_rate_max is not None:
            if self.u_rate_min.shape != (1, self.n_u):
                self.u_rate_min = self.u_rate_min.reshape(1, self.n_u)
            if self.u_rate_max.shape != (1, self.n_u):
                self.u_rate_max = self.u_rate_max.reshape(1, self.n_u)
            for i in range(self.m):
                for k in range(self.K - 1):
                    self.ocp.subject_to(
                        self.u_rate_min <= self.u[i][k + 1, :] - self.u[i][k, :]
                    )
                    self.constraints.append(f"input rate constraints min i:{i} k:{k}")
                    self.ocp.subject_to(
                        self.u[i][k + 1, :] - self.u[i][k, :] <= self.u_rate_max
                    )
                    self.constraints.append(f"input rate constraints max i:{i} k:{k}")

        # Total input constraints
        if self.u_tot_max is not None:
            for i in range(self.m):
                for k in range(self.K):
                    self.ocp.subject_to(ca.norm_1(self.u[i][k, :]) <= self.u_tot_max)
                    self.constraints.append(f"total input constraints i:{i} k:{k}")

        # Collision avoidance constraints
        if self.d_min is not None:
            for i in range(self.m):
                for j in range(self.m):
                    if i == j:
                        continue  # skip self-collision
                    for k in range(self.K + 1):
                        dist = (self.x[i][k, 0] - self.x[j][k, 0]) ** 2 + (
                            self.x[i][k, 1] - self.x[j][k, 1]
                        ) ** 2
                        self.ocp.subject_to(dist >= self.d_min**2)
                        self.constraints.append(
                            f"collision avoidance i:{i} j:{j} k:{k}"
                        )

        # Cost function
        self.cost_function = 0

        # Goal tracking cost
        for i in range(self.m):
            for k in range(1, self.K + 1):
                self.cost_function += self.alpha_g * (
                    (self.x[i][k, 0] - self.x_goal[0, i]) ** 2
                    + (self.x[i][k, 1] - self.x_goal[1, i]) ** 2
                )

        # Control input cost
        for i in range(self.m):
            for k in range(self.K):
                self.cost_function += self.alpha_u * (
                    self.u[i][k, :] @ self.R @ self.u[i][k, :].T
                )

        # Winding cost + winding constraints for guarantees on specification tracking
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue  # skip self-winding

                # Get weight for agent i w.r.t. agent j
                alpha_w_ij: float = self.alpha_w[i, j]

                # Compute winding number w.r.t. j at the end of prediction horizon
                w = self.w_curr[i, j]
                for k in range(1, self.K + 1):
                    theta: ca.SX | ca.MX = ca.atan2(
                        self.x[j][k, 1] - self.x[i][k, 1],
                        self.x[j][k, 0] - self.x[i][k, 0],
                    )
                    theta_prev: ca.SX | ca.MX = ca.atan2(
                        self.x[j][k - 1, 1] - self.x[i][k - 1, 1],
                        self.x[j][k - 1, 0] - self.x[i][k - 1, 0],
                    )
                    w += 1 / (2 * np.pi) * self.angle_diff(theta, theta_prev)

                    # Add winding cost to the total cost function
                    self.cost_function += alpha_w_ij * (self.w_target[k][i, j] - w) ** 2

                    # Add winding constraint
                    self.ocp.subject_to(
                        ca.fabs(w - self.w_target[k][i, j]) < self.w_epsilon
                    )

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
        if w_target.shape != (self.K + 1, self.m, self.m):
            raise ValueError(
                f"w_target must have shape ({self.K + 1}, {self.m}, {self.m}), "
                f"but got {w_target.shape}."
            )
        if sol_prev is not None and not isinstance(sol_prev, ca.OptiSol):
            raise ValueError(
                "sol_prev must be of type casadi.OptiSol or None, but got "
                f"{type(sol_prev)}."
            )

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
        for k in range(self.K + 1):
            self.ocp.set_value(self.w_target[k], w_target[k, :, :])

        # Warm start
        if use_warm_start is True and sol_prev is not None:
            self.ocp.set_initial(sol_prev.value_variables())
        if use_warm_start is True and sol_prev is None and self.sol is not None:
            self.ocp.set_initial(self.sol.value_variables())
        else:
            # always warm start with x_0 to avoid numerical errors
            # See: https://github.com/casadi/casadi/discussions/3539
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-"NaN-detected"in-my-optimization%3F  # pylint: disable=line-too-long
            for i in range(self.m):
                for k in range(self.K + 1):
                    self.ocp.set_initial(self.x[i][k, :], x_0[:, i])

        # Solve OCP and return solution object
        try:
            sol = self.ocp.solve()
        except RuntimeError as e:
            if self.ocp.debug.stats()["return_status"] in [
                "Maximum_CpuTime_Exceeded",
                "Maximum_Iterations_Exceeded",
            ]:
                sol = self.ocp.debug  # FIXME solution attributes must be assigned
            else:
                g_val = self.ocp.debug.value(self.ocp.g)
                lb_g = np.array(self.ocp.debug.value(self.ocp.lbg)).flatten()
                ub_g = np.array((self.ocp.debug.value(self.ocp.ubg))).flatten()
                violation = np.maximum(g_val - ub_g, 0) + np.maximum(lb_g - g_val, 0)
                violation_idx = np.argsort(violation)[-10:]  # 10 largest violations
                for idx in violation_idx:
                    print(
                        f"Constraint {idx}: "
                        f"{self.ocp.debug.g_describe(idx)[158:-1]}."
                        f"violation={violation[idx]:.4e}, "
                        f"g={g_val[idx]:.4e}, "
                        f"lb={lb_g[idx]:.4e}, ub={ub_g[idx]:.4e}"
                    )
                raise e

        # Return the solution object
        return sol

    def check_cost(self) -> tuple[float, float, float, float]:
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

        # Extract the OCP parameters and solution values
        x = [self.sol.value(x_i) for x_i in self.x]
        x_goal = self.sol.value(self.x_goal)
        w_curr = self.sol.value(self.w_curr)
        w_target = [self.sol.value(w_target_k) for w_target_k in self.w_target]
        u = [self.sol.value(u_i) for u_i in self.u]

        # Goal tracking cost
        goal_cost = 0
        for i in range(self.m):
            for k in range(1, self.K + 1):
                goal_cost += self.ocp.value(self.alpha_g) * (
                    (x[i][k, 0] - x_goal[0, i]) ** 2 + (x[i][k, 1] - x_goal[1, i]) ** 2
                )

        # Control input cost
        control_cost = 0
        for i in range(self.m):
            for k in range(self.K):
                control_cost += self.alpha_u * (u[i][k, :] @ self.R @ u[i][k, :].T)

        # Winding cost
        winding_cost = 0
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue  # skip self-winding

                # Get weight for agent i w.r.t. agent j
                alpha_w_ij = self.ocp.value(self.alpha_w[i, j])

                # Compute winding number of i w.r.t. j at end of prediction horizon
                w = w_curr[i, j]
                for k in range(1, self.K + 1):
                    theta = np.arctan2(
                        x[j][k, 1] - x[i][k, 1],
                        x[j][k, 0] - x[i][k, 0],
                    )
                    theta_prev = np.arctan2(
                        x[j][k - 1, 1] - x[i][k - 1, 1],
                        x[j][k - 1, 0] - x[i][k - 1, 0],
                    )
                    delta_theta = invariants.angle_diff(theta, theta_prev)
                    w += 1 / (2 * np.pi) * delta_theta

                    # Add cost of step k for agent i w.r.t. j to the total winding cost
                    winding_cost += alpha_w_ij * (w_target[k][i, j] - w) ** 2

        # Total cost
        cost = goal_cost + control_cost + winding_cost

        # Cost validation
        if not np.isclose(cost, self.sol.value(self.cost_function), atol=1e-4):
            print(
                f"Cost check failed: computed cost {cost:.4f} does not match "
                f"cost from solution {self.sol.value(self.cost_function):.4f} "
                f"(error: {abs(cost - self.sol.value(self.cost_function)):.4e})."
            )

        return cost, goal_cost, control_cost, winding_cost
