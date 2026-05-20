import casadi as ca
import numpy as np

from braid_controller.core.mpc import MPC
from braid_controller.utils import invariants


class DistributedMPC(MPC):

    def __init__(self, dynamics: str = "single_integrator") -> None:
        super().__init__(dynamics=dynamics)
        self.architecture: str = "distributed"
        self.collision_avoidance: str = "convex"  # {convex, nonconvex}
        self.slack_collision_constraints: bool = False  # slack collision constraints
        self.slack_state_constraints: bool = False  # slack state constraints
        self.solver_options["print_level"] = 0
        self.solver_options["max_wall_time"] = 10.0  # [s]
        self.solver_options["max_cpu_time"] = 10.0  # [s]

        # Optimization problem parameters
        # In the distributed MPC case, each agent's local OCP problem includes the
        # predicted trajectories (optimizers at the previous time step) of the other
        # agents, which are shared among the agents at each time step.
        # x_pred is a list of length m-1, where each element is a (K+1, n_x) array
        # representing the predicted trajectory of another agent over the prediction
        # horizon. The predicted trajectory of the ego agent itself is represented by
        # the x_prev variable. In both x_pred and x_prev the last element is repeated
        # twice to get the shape (K+1, n_x).
        self.x_pred: list[ca.Opti.parameter]  # m-1 arrays of shape (K+1, n_x)
        self.x_prev: ca.Opti.parameter  # optimizer of ego agent at previous time step

        # Variables for slack collision avoidance constraints
        self.s_coll: ca.Opti.variable
        self.s_state: ca.Opti.variable
        self.alpha_s_1: float
        self.alpha_s_2: float

    def set_alpha_w(self, alpha_w: np.ndarray) -> None:
        """
        Set the winding cost weight.

        Args:
            alpha_w (np.ndarray): winding cost weight matrix of shape (m-1, ).

        Returns:
            None

        Raises:
            ValueError: if alpha_w is a numpy array with incorrect shape
            TypeError: if alpha_w is not a scalar or a numpy array
        """
        if not isinstance(alpha_w, np.ndarray):
            raise TypeError("alpha_w must be a numpy array.")
        if not alpha_w.shape == (self.m - 1,):
            raise ValueError(
                f"alpha_w must have shape ({self.m - 1}, ), "
                f"but got {alpha_w.shape}."
            )
        self.ocp.set_value(self.alpha_w, alpha_w)

    def _initialize_ocp(self) -> None:
        """
        Initialize the local OCP problem for the distributed MPC architecture.
        """
        # Initialize optimization variables
        self.x = self.ocp.variable(self.K + 1, self.n_x)  # state trajectory (K+1, n_x)
        self.u = self.ocp.variable(self.K, self.n_u)  # control input (K, n_u)

        # slack variables for collision avoidance and related weights
        if self.slack_collision_constraints is True:
            self.s_coll = self.ocp.variable(self.m - 1, self.K + 1)
        if self.slack_state_constraints is True:
            self.s_state = self.ocp.variable(self.K + 1, self.n_x_pos)
        self.alpha_s_1 = 1e3
        self.alpha_s_2 = 1e3

        # Initialize OCP parameters
        # NOTE: in the distributed case, the list of predicted states of the agents,
        # as well as the arrays of current and goal winding numbers, must not include
        # the ego agent itself. The ego agent's information must be removed before
        # passing the inputs to the MPC controller.
        self.x_0 = self.ocp.parameter(1, self.n_x)
        self.x_goal = self.ocp.parameter(1, self.n_x)
        self.w_curr = self.ocp.parameter(self.m - 1)  # current winding number (m-1, )
        self.w_target = self.ocp.parameter(
            self.K + 1, self.m - 1
        )  # target winding numbers over the horizon (K+1, m-1)
        self.x_pred = [
            self.ocp.parameter(self.K + 1, self.n_x) for _ in range(self.m - 1)
        ]  # predicted state of other agents (m-1 arrays of shape (K+1, n_x))
        self.x_prev = self.ocp.parameter(self.K + 1, self.n_x)  # ego prediction at k-1
        self.alpha_g = self.ocp.parameter()  # scalar
        self.alpha_w = self.ocp.parameter(self.m - 1)  # winding weight (m-1, )

        # Constraints
        # Initial state constraint
        self.ocp.subject_to(self.x[0, :] == self.x_0)

        # Dynamics
        for k in range(self.K):
            x_next = self.x[k, :] + self.dxdt(self.x[k, :], self.u[k, :]) * self.dt
            self.ocp.subject_to(self.x[k + 1, :] == x_next)

        # State constraints
        if self.x_min is not None and self.x_max is not None:
            if self.x_min.shape != (1, self.n_x_pos):
                self.x_min = self.x_min.reshape(1, self.n_x_pos)
            if self.x_max.shape != (1, self.n_x_pos):
                self.x_max = self.x_max.reshape(1, self.n_x_pos)
            for k in range(self.K + 1):
                if self.slack_state_constraints is True:
                    # Add slack variable to the constraint
                    self.ocp.subject_to(
                        self.x_min - self.s_state[k, : self.n_x_pos]
                        <= self.x[k, : self.n_x_pos]
                    )
                    self.ocp.subject_to(
                        self.x[k, : self.n_x_pos]
                        <= self.x_max + self.s_state[k, : self.n_x_pos]
                    )
                else:
                    self.ocp.subject_to(self.x_min <= self.x[k, : self.n_x_pos])
                    self.ocp.subject_to(self.x[k, : self.n_x_pos] <= self.x_max)

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            if self.u_min.shape != (1, self.n_u):
                self.u_min = self.u_min.reshape(1, self.n_u)
            if self.u_max.shape != (1, self.n_u):
                self.u_max = self.u_max.reshape(1, self.n_u)
            for k in range(self.K):
                self.ocp.subject_to(self.u_min <= self.u[k, :])
                self.ocp.subject_to(self.u[k, :] <= self.u_max)

        # Input rate constraints
        # NOTE: the previous u (i.e., the one applied at the previous time step) should
        # also be taken into account to constraint u[0], but for simplicity we only
        # constraint the rate between the optimization variables (at least for now).
        if self.u_rate_min is not None and self.u_rate_max is not None:
            if self.u_rate_min.shape != (1, self.n_u):
                self.u_rate_min = self.u_rate_min.reshape(1, self.n_u)
            if self.u_rate_max.shape != (1, self.n_u):
                self.u_rate_max = self.u_rate_max.reshape(1, self.n_u)
            for k in range(self.K - 1):
                self.ocp.subject_to(self.u_rate_min <= self.u[k + 1, :] - self.u[k, :])
                self.ocp.subject_to(self.u[k + 1, :] - self.u[k, :] <= self.u_rate_max)

        # Total input constraints
        if self.u_tot_max is not None:
            for k in range(self.K):
                self.ocp.subject_to(ca.norm_1(self.u[k, :]) <= self.u_tot_max)

        # Collision avoidance constraints
        if self.collision_avoidance == "convex":
            # The collision avoidance constraints are cast as a sequence of convex
            # constraints that enforce a minimum distance between the trajectory of the
            # ego agent and the predicted trajectories of the other agents. In order to
            # guarantee recursive feasibility, the linear constraint is based on the
            # predicted trajectory of the ego agent at the previous time step, so that
            # the constraint is the same for both agents i (ego) and j involved. In
            # order to guarantee recursive feasibility, this constraint is aslo paired
            # with a terminal constraint on the state x(N|k), which is trivial for a
            # single-integrator and unicycle dynamics but must be accounted in general.
            # NOTE: the distance constraint is only computed w.r.t. other agents, and
            # not w.r.t. itself. Therefore only m-1 constraints are considered here.
            # NOTE: When using slack==True the d_min distance is assumed to be already
            # buffered to account for the possible violations of the constraint due to
            # it being only a soft constraint.

            # Start by computing the safety distance d (>= self.d_min)
            d: float
            if self.dynamics == "single_integrator":
                if self.u_max is not None:
                    v_max = np.sqrt(self.u_max[0, 0] ** 2 + self.u_max[0, 1] ** 2)
                    d = np.sqrt(self.d_min**2 + (v_max * self.dt) ** 2)
                else:
                    d = self.d_min
            elif self.dynamics == "unicycle":
                if self.u_max is not None:
                    d = np.sqrt(self.d_min**2 + self.u_max[0, 0] * self.dt**2)
                else:
                    d = self.d_min

            # Compute separating hyperplane between predicted trajectories  of i and j
            for j in range(self.m - 1):
                for k in range(self.K + 1):
                    a_ij = (
                        self.x_prev[k, : self.n_x_pos]
                        - self.x_pred[j][k, : self.n_x_pos]
                    ) / ca.norm_2(
                        self.x_prev[k, : self.n_x_pos]
                        - self.x_pred[j][k, : self.n_x_pos]
                    )
                    b_ij = (
                        ca.dot(
                            a_ij,
                            (
                                self.x_prev[k, : self.n_x_pos]
                                + self.x_pred[j][k, : self.n_x_pos]
                            ),
                        )
                        / 2
                        + d / 2
                    )
                    if self.slack_collision_constraints is True:
                        # Add slack variable to the constraint
                        self.ocp.subject_to(
                            ca.dot(a_ij, self.x[k, : self.n_x_pos])
                            >= b_ij - self.s_coll[j, k]
                        )
                    else:
                        self.ocp.subject_to(
                            ca.dot(a_ij, self.x[k, : self.n_x_pos]) >= b_ij
                        )

        elif self.collision_avoidance == "nonconvex":
            # Uses a nonconvex constraint (quadratic constraint for circular agents).
            # In general, the use ofa nonconvex constraint can lead to infeasibility
            # of the OCP. To prevent this, it is recommended to use a slack variable to
            # promote a minimum distance between the ego agent and the predicted
            # trajectories of the other agents, while keeping the problem feasible.
            # The slack variable is penalized in the cost function.
            # NOTE: When using slack==True the d_min distance is assumed to be already
            # buffered to account for the possible violations of the constraint due to
            # it being only a soft constraint.
            for j in range(self.m - 1):
                for k in range(self.K + 1):
                    dist_sq = (self.x[k, 0] - self.x_pred[j][k, 0]) ** 2 + (
                        self.x[k, 1] - self.x_pred[j][k, 1]
                    ) ** 2
                    if self.slack_collision_constraints is True:
                        self.ocp.subject_to(
                            dist_sq + self.s_coll[j, k] >= self.d_min**2
                        )
                    else:
                        self.ocp.subject_to(dist_sq >= self.d_min**2)
        else:
            raise ValueError(
                f"Invalid collision avoidance method: {self.collision_avoidance}."
            )

        # Slack variable constraint (to ensure s is non-negative)
        if self.slack_collision_constraints is True:
            self.ocp.subject_to(ca.vec(self.s_coll) >= 0)
        if self.slack_state_constraints is True:
            self.ocp.subject_to(ca.vec(self.s_state) >= 0)

        # Cost function
        self.cost_function = 0

        # Goal tracking cost (terminal cost)
        for k in range(1, self.K + 1):
            self.cost_function += self.alpha_g * (
                (self.x[k, 0] - self.x_goal[0]) ** 2
                + (self.x[k, 1] - self.x_goal[1]) ** 2
            )

        # Control input cost
        for k in range(self.K):
            self.cost_function += self.alpha_u * (
                self.u[k, :] @ self.R @ self.u[k, :].T
            )

        # Slack variable cost for state constraints (if enabled)
        # Slack penalty is (L1 + L2). L1 gives exact-penalty behavior; L2 helps the QP.
        if self.slack_collision_constraints is True:
            self.cost_function += self.alpha_s_1 * ca.sum1(ca.vec(self.s_coll))
            self.cost_function += self.alpha_s_2 * ca.sumsqr(self.s_coll)
        if self.slack_state_constraints is True:
            self.cost_function += self.alpha_s_1 * ca.sum1(ca.vec(self.s_state))
            self.cost_function += self.alpha_s_2 * ca.sumsqr(self.s_state)

        # Winding cost + winding constraints for guarantees on specification tracking
        # NOTE: the winding cost is only computed w.r.t. other agents, and not w.r.t.
        # itself, since the winding number w.r.t. itself is always 0. Therefore only m-1
        # terms are summed in the winding cost.
        for j in range(self.m - 1):

            # Get weight for agent j
            alpha_w_j = self.alpha_w[j]

            # Compute winding number w.r.t. j at the end of prediction horizon
            w = self.w_curr[j]
            for k in range(1, self.K + 1):
                theta: ca.MX = ca.atan2(
                    self.x_pred[j][k, 1] - self.x[k, 1],
                    self.x_pred[j][k, 0] - self.x[k, 0],
                )
                theta_prev: ca.MX = ca.atan2(
                    self.x_pred[j][k - 1, 1] - self.x[k - 1, 1],
                    self.x_pred[j][k - 1, 0] - self.x[k - 1, 0],
                )
                w += 1 / (2 * np.pi) * self.angle_diff(theta, theta_prev)

                # Add winding cost to the total cost function
                self.cost_function += alpha_w_j * (self.w_target[k, j] - w) ** 2

                # Add winding constraint
                self.ocp.subject_to(ca.fabs(w - self.w_target[k, j]) < self.w_epsilon)

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
        Solve the MPC problem for the distributed MPC architecture.

        Kwargs:
            x_pred (list[np.ndarray] | np.ndarray): predicted trajectories of the other
                agents over the prediction horizon. This can be either a list of length
                m-1, where each element is a (K+1, n_x) array representing the predicted
                trajectory of another agent, or a single array of shape (K+1, n_x, m-1)
                containing the predicted trajectories of all other agents.
            x_prev (np.ndarray): predicted trajectory of the ego agent at the previous
                time step, of shape (K+1, n_x).

        Raises:
            ValueError: if the required kwargs are not provided.
            ValueError: if the input arrays have incorrect shapes or types.
            TypeError: if the input arrays have incorrect types.
        """
        # Validate inputs
        if x_0.shape != (self.n_x,):
            raise ValueError(f"x_0 must have shape ({self.n_x},), but got {x_0.shape}.")
        if x_goal.shape != (self.n_x,):
            raise ValueError(
                f"x_goal must have shape ({self.n_x},), but got {x_goal.shape}."
            )
        if w_curr.shape != (self.m - 1,):
            raise ValueError(
                f"w_curr must have shape ({self.m - 1},), but got {w_curr.shape}."
            )
        if w_target.shape != (self.K + 1, self.m - 1):
            raise ValueError(
                f"w_target must have shape ({self.K + 1}, {self.m - 1}), "
                f"but got {w_target.shape}."
            )
        if sol_prev is not None and not isinstance(sol_prev, ca.OptiSol):
            raise ValueError(
                "sol_prev must be of type casadi.OptiSol or None, but got "
                f"{type(sol_prev)}."
            )

        # Parse kwargs
        x_pred: list[np.ndarray] | np.ndarray = kwargs.get("x_pred", None)
        if x_pred is None:
            raise ValueError("x_pred must be provided as a kwarg parameter.")
        elif isinstance(x_pred, list):
            if len(x_pred) != self.m - 1:
                raise ValueError(
                    f"x_pred must be a list of length {self.m - 1}, but got "
                    f"{len(x_pred)}."
                )
            for j, x_pred_j in enumerate(x_pred):
                if x_pred_j.shape != (self.K + 1, self.n_x):
                    raise ValueError(
                        f"x_pred[{j}] must have shape ({self.K + 1}, {self.n_x}), "
                        f"got {x_pred_j.shape} instead."
                    )
            x_pred = np.array(x_pred)  # convert to numpy array for ease of indexing
        elif isinstance(x_pred, np.ndarray):
            if x_pred.shape != (self.K + 1, self.n_x, self.m - 1):
                raise ValueError(
                    f"x_pred must have shape ({self.K + 1}, {self.n_x}, {self.m - 1}), "
                    f"but got {x_pred.shape}."
                )
        else:
            raise TypeError(
                f"x_pred must be either a list of numpy arrays or a single numpy "
                f"array, got {type(x_pred)} instead."
            )
        x_prev: np.ndarray = kwargs.get("x_prev", None)
        if x_prev is None:
            raise ValueError("x_prev must be provided as a kwarg parameter.")
        elif not isinstance(x_prev, np.ndarray):
            raise TypeError(
                f"x_prev must be a numpy array, but got {type(x_prev)} instead."
            )
        elif x_prev.shape != (self.K + 1, self.n_x):
            raise ValueError(
                f"x_prev must have shape ({self.K + 1}, {self.n_x}), but got "
                f"{x_prev.shape}."
            )

        # Set parameters in OCP
        self.ocp.set_value(self.x_0, x_0)
        self.ocp.set_value(self.x_goal, x_goal)
        self.ocp.set_value(self.w_curr, w_curr)
        self.ocp.set_value(self.w_target, w_target)
        self.ocp.set_value(self.x_prev, x_prev)
        for j in range(self.m - 1):
            self.ocp.set_value(self.x_pred[j], x_pred[:, :, j])

        # Warm start
        # NOTE: in distributed mpc the initial guess must be passed manually, as the
        # solution stored in self.sol corresponds to a different agent and cannot be
        # used. If sol_prev is not provided, the mpc defaults to warm starting with the
        # initial position.
        if use_warm_start is True and sol_prev is not None:
            self.ocp.set_initial(sol_prev.value_variables())
        else:
            # always warm start with x_0 to avoid numerical errors
            # See: https://github.com/casadi/casadi/discussions/3539
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-"NaN-detected"in-my-optimization%3F  # pylint: disable=line-too-long
            for k in range(self.K + 1):
                self.ocp.set_initial(self.x[k, :], x_0)

        # Always warm start slack variable to 0 if used
        if self.slack_collision_constraints is True:
            self.ocp.set_initial(self.s_coll, 0)
        if self.slack_state_constraints is True:
            self.ocp.set_initial(self.s_state, 0)

        # Solve OCP and return solution object
        sol = self.ocp.solve()
        return sol

    def check_cost(self, **kwargs) -> tuple[float, float, float, float]:
        """
        Compute the cost function value for the current solution. This method is mainly
        intended for debugging and analysis purposes, as the cost value is already
        computed by the solver and can be extracted from the solution object.

        Note: this method must be called immediately after solving the OCP as the
            solution gets overwritten upon switching to the next agent. To

        Args:
            idx (int, optional): index of the agent for which to compute the cost.
                This is only used to show the agent idx in the output print statement.

        Returns:
            cost (float): value of the cost function for the current solution
            goal_cost (float): contribution of the goal tracking term to the total cost
            control_cost (float): contribution of control input term to the total cost
            winding_cost (float): contribution of the winding term to the total cost

        Raises:
            RuntimeError: if the OCP has not been solved yet (no solution available).
            ValueError: if the sum of the cost components does not match the total cost.
        """
        # Parse kwargs
        idx: int | None = kwargs.get("idx", None)

        # Check if OCP is initialized
        if self.sol is None:
            raise RuntimeError(
                "No solution available. Please solve the OCP before calling "
                "check_cost()."
            )

        # Extract the OCP parameters and solution values
        x = self.sol.value(self.x)
        x_goal = self.sol.value(self.x_goal)
        x_pred = [self.sol.value(x_pred_j) for x_pred_j in self.x_pred]
        w_curr = self.sol.value(self.w_curr)
        w_target = self.sol.value(self.w_target)
        u = self.sol.value(self.u)

        # Goal tracking cost
        goal_cost = 0
        for k in range(1, self.K + 1):
            goal_cost += self.ocp.value(self.alpha_g) * (
                (x[k, 0] - x_goal[0]) ** 2 + (x[k, 1] - x_goal[1]) ** 2
            )

        # Control input cost
        control_cost = 0
        for k in range(self.K):
            control_cost += self.alpha_u * (u[k, :] @ self.R @ u[k, :].T)

        # Winding cost
        winding_cost = 0
        for j in range(self.m - 1):

            # Get weight for agent j
            alpha_w_j = self.ocp.value(self.alpha_w[j])

            # Compute winding number w.r.t. j at the end of prediction horizon
            w: float = w_curr[j] if isinstance(w_curr, np.ndarray) else w_curr
            if isinstance(w_target[:, j], np.ndarray):
                w_target_j = w_target[:, j]
            else:
                w_target_j = w_target
            for k in range(1, self.K + 1):
                theta: float = np.arctan2(
                    x_pred[j][k, 1] - x[k, 1],
                    x_pred[j][k, 0] - x[k, 0],
                )
                theta_prev: float = np.arctan2(
                    x_pred[j][k - 1, 1] - x[k - 1, 1],
                    x_pred[j][k - 1, 0] - x[k - 1, 0],
                )
                w += 1 / (2 * np.pi) * invariants.angle_diff(theta, theta_prev)

                # Cumulate winding costs
                winding_cost += alpha_w_j * (w_target_j[k] - w) ** 2

        # Slack variable cost
        slack_cost = 0
        n_slacks = 0
        if self.slack_state_constraints is True:
            s_coll = self.sol.value(self.s_coll)  # shape (m-1, K+1)
            slack_cost += self.alpha_s_1 * np.sum(s_coll) + self.alpha_s_2 * np.sum(
                s_coll**2
            )
            n_slacks += (self.m - 1) * (self.K + 1)
        if self.slack_state_constraints is True:
            s_state = self.sol.value(self.s_state)  # shape (K+1, n_x_pos)
            slack_cost += self.alpha_s_1 * np.sum(s_state) + self.alpha_s_2 * np.sum(
                s_state**2
            )
            n_slacks += (self.K + 1) * self.n_x_pos
            if (
                slack_cost >= self.alpha_s_1 * n_slacks * (-9.09e-9)
                and self.debug is True
            ):
                print(f"Slack variable was activated: slack cost: {slack_cost}")

        # Total cost
        cost = goal_cost + control_cost + winding_cost + slack_cost

        # Check if the sum of the cost components matches the total cost
        if (
            not np.isclose(cost, self.sol.value(self.cost_function), atol=1e-6)
            and self.debug is True
        ):
            if idx is not None:
                print(f"Cost check failed for agent {idx}: ", end="")
            else:
                print("Cost check failed: ", end="")
            print(
                f"computed cost {cost:.4f} does not match "
                f"cost from solution {self.sol.value(self.cost_function):.4f} "
                f"(error: {abs(cost - self.sol.value(self.cost_function)):.4e})."
            )

        # NOTE: slack cost not returned for consistency with case where no slack used
        return cost, goal_cost, control_cost, winding_cost
