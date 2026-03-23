import casadi as ca
import numpy as np

from utils import invariants


class MPC:

    def __init__(self) -> None:
        """
        Initialize the MPC controller.

        Args:
            agent_id (int): Unique identifier for the agent.

        Returns:
            None
        """
        # MPC parameters
        self.dt: float | None = None  # time step
        self.K: int | None = None  # prediction horizon
        self.m: int | None = None  # number of agents

        # Optimization problem
        self.ocp: ca.Opti = ca.Opti()
        self.ocp_ready: bool = False  # flag to check if the OCP is initialized
        self.ocp_options: dict = {
            "expand": True,
            "error_on_fail": False,  # suppress compiler output
            "print_time": 0,  # suppress compiler output
            "verbose": False,  # suppress compiler output
            "record_time": True,  # record time statistics
        }
        self.solver: str = "ipopt"
        self.solver_options: dict = {
            "max_iter": 10_000,
            "max_wall_time": 120.0,  # [s]
            "max_cpu_time": 60.0,  # [s]
            "print_level": 0,  # 0-5 (0 = silent, 5 = verbose)
            "tol": 1e-6,  # optimality tolerance
            "acceptable_tol": 1e-4,  # tolerance for early termination
            "linear_solver": "mumps",  # recommended for most problems
        }
        self.sol: ca.OptiSol | None = None  # solution of the OCP

        # Optimization variables
        self.n_u: int = 2  # number of control inputs
        self.n_x: int = 2  # number of states
        self.u: ca.Opti.variable  # control input (n_u, K)
        self.x: ca.Opti.variable  # state trajectory (n_x, K+1)

        # Optimization problem parameters
        # NOTE: the list of predicted states of the agents, as well as the arrays of
        # winding numbers, must not include the ego agent itself. The ego agent's info
        # must be removed before passing the inputs to the MPC controller.
        self.x_0: ca.Opti.parameter  # initial state (n_x, )
        self.x_goal: ca.Opti.parameter  # goal state (n_x, )
        self.x_pred: list[ca.Opti.parameter]  # agents' predicted states (m-1, n_x, K+1)
        self.w_curr: ca.Opti.parameter  # current winding number w.r.t. agents (m-1, )
        self.w_target: ca.Opti.parameter  # target winding number w.r.t. agents (m-1, )
        self.cost_function: ca.Opti.variable  # cost function

        # Cost function weights and matrices
        self.alpha_u: float | None = None  # control input cost weight
        self.alpha_goal: float | None = None  # goal cost weight
        self.alpha_w: np.ndarray | float | None = None  # winding cost weight (m-1, )
        self.R: np.ndarray | None = None  # control input cost matrix

        # Constraints
        self.u_min: np.ndarray | None = None  # (n_u, )
        self.u_max: np.ndarray | None = None  # (n_u, )
        self.u_rate_min: np.ndarray | None = None  # (n_u, )
        self.u_rate_max: np.ndarray | None = None  # (n_u, )
        self.u_tot_max: float | None = None  # maximum total control input
        self.x_min: np.ndarray | None = None  # (n_x, )
        self.x_max: np.ndarray | None = None  # (n_x, )
        self.d_min: float | None = None  # minimum distance between agents

    def __call__(
        self,
        x_0: np.ndarray,
        x_goal: np.ndarray,
        x_pred: list[np.ndarray],
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol: ca.OptiSol | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Call the MPC controller to solve the NMPC problem. See the `solve` method for
        details on the arguments and return values.
        """
        u, x_pred, cost, time = self.solve(
            x_0,
            x_goal,
            x_pred,
            w_curr,
            w_target,
            use_warm_start,
            sol,
        )
        return u, x_pred, cost, time

    @staticmethod
    def angle_diff(
        angle_1: float | ca.SX | ca.MX,
        angle_2: float | ca.SX | ca.MX,
    ) -> float | ca.SX | ca.MX:
        """
        Helper function to avoid numerical issues in modulo arithmetic.

        Args:
            angle_1 (float | ca.SX | ca.MX): First angle in radians.
            angle_2 (float | ca.SX | ca.MX): Second angle in radians.

        Returns:
            float | ca.SX | ca.MX: The difference between the two angles, normalized to
                the range (-pi, pi].
        """
        return ca.atan2(ca.sin(angle_1 - angle_2), ca.cos(angle_1 - angle_2))

    def initialize_ocp(self) -> None:
        """
        Initialize the optimal control problem (OCP) for the MPC controller. This method
        should be called before calling the `solve` method for the first time.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: if the OCP has already been initialized.
        """
        # Check if the OCP has already been initialized
        if self.ocp_ready:
            raise RuntimeError("The OCP has already been initialized.")

        # Initialize optimization variables
        self.x = self.ocp.variable(self.n_x, self.K + 1)  # state trajectory (n_x, K+1)
        self.u = self.ocp.variable(self.n_u, self.K)  # control input (n_u, K)

        # Initialize OCP parameters
        self.x_0 = self.ocp.parameter(self.n_x)
        self.x_goal = self.ocp.parameter(self.n_x)
        self.w_curr = self.ocp.parameter(self.m, self.n_x)
        self.w_target = self.ocp.parameter(self.m, self.n_x)
        self.x_pred = [self.ocp.parameter(self.n_x, self.K + 1) for _ in range(self.m)]

        # Constraints
        # Initial state constraint
        self.ocp.subject_to(self.x[:, 0] == self.x_0)

        # Dynamics
        for k in range(self.K):
            x_next = self.x[:, k] + self._dxdt(self.u[:, k]) * self.dt
            self.ocp.subject_to(self.x[:, k + 1] == x_next)

        # State constraints
        if self.x_min is not None and self.x_max is not None:
            for k in range(self.K + 1):
                self.ocp.subject_to(self.x_min <= self.x[:, k])
                self.ocp.subject_to(self.x[:, k] <= self.x_max)

        # Input constraints
        if self.u_min is not None and self.u_max is not None:
            for k in range(self.K):
                self.ocp.subject_to(self.u_min <= self.u[:, k])
                self.ocp.subject_to(self.u[:, k] <= self.u_max)

        # Input rate constraints
        if self.u_rate_min is not None and self.u_rate_max is not None:
            for k in range(self.K - 1):
                self.ocp.subject_to(self.u_rate_min <= self.u[:, k + 1] - self.u[:, k])
                self.ocp.subject_to(self.u[:, k + 1] - self.u[:, k] <= self.u_rate_max)

        # Total input constraints
        if self.u_tot_max is not None:
            for k in range(self.K):
                self.ocp.subject_to(ca.norm_1(self.u[:, k]) <= self.u_tot_max)

        # Collision avoidance constraints
        # NOTE: the distance constraint is only computed w.r.t. other agents, and not
        # w.r.t. itself. Therefore only m-1 constraints are considered here.
        if self.d_min is not None:
            for j in range(self.m - 1):
                for k in range(self.K + 1):
                    dist = ca.norm_2(self.x[:, k] - self.x_pred[j][:, k])
                    self.ocp.subject_to(dist >= self.d_min)

        # Cost function
        self.cost_function = 0

        # Goal tracking cost
        if self.alpha_goal is not None and self.alpha_goal > 0:
            for k in range(self.K + 1):
                self.cost_function += self.alpha_goal * (
                    (self.x[0, k] - self.x_goal[0]) ** 2
                    + (self.x[1, k] - self.x_goal[1]) ** 2
                )

        # Control input cost
        if self.alpha_u is not None and self.alpha_u > 0:
            for k in range(self.K):
                self.cost_function += self.alpha_u * (
                    self.u[:, k].T @ self.R @ self.u[:, k]
                )

        # Winding cost
        # NOTE: the winding cost is only computed w.r.t. other agents, and not w.r.t.
        # itself, since the winding number w.r.t. itself is always 0. Therefore only m-1
        # terms are summed in the winding cost.
        if self.alpha_w is not None and np.any(self.alpha_w > 0):
            for j in range(self.m - 1):

                # Get weight for agent j
                if isinstance(self.alpha_w, np.ndarray):
                    self.alpha_w: np.ndarray  # to avoid unsubscriptable-object
                    alpha_w_j = self.alpha_w[j]
                else:
                    alpha_w_j = self.alpha_w

                # Compute winding number w.r.t. j at the end of prediction horizon
                w = self.w_curr[j]
                for k in range(1, self.K + 1):
                    theta: ca.SX | ca.MX = ca.atan2(
                        self.x[1, k] - self.x_pred[j][1, k],
                        self.x[0, k] - self.x_pred[j][0, k],
                    )
                    theta_prev: ca.SX | ca.MX = ca.atan2(
                        self.x[1, k - 1] - self.x_pred[j][1, k - 1],
                        self.x[0, k - 1] - self.x_pred[j][0, k - 1],
                    )
                    w += 1 / (2 * np.pi) * self.angle_diff(theta, theta_prev)

                # Add winding cost to the total cost function
                self.cost_function += alpha_w_j * (self.w_target[j] - w) ** 2

        # Define the objective
        self.ocp.minimize(self.cost_function)

        # Set the solver options
        self.ocp.solver(self.solver, self.ocp_options, self.solver_options)

        # Set ready flag to true
        self.ocp_ready = True

    def _dxdt(
        self,
        u: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Continuous-time dynamics function for the MPC controller. To be integrated
        numerically within the MPC prediction model. The dynamics are defined as:
            dxdt = f(x, u)

        The state is structured as:
            x = [x, y]^T

        The control input is structured as:
            u = [u_x, u_y]^T

        Args:
            x (ca.SX or ca.MX): State vector (n_x, ).
            u (ca.SX or ca.MX): Control input vector (n_u, ).

        Returns:
            dxdt (ca.SX or ca.MX): Time derivative of the state vector (n_x, ).
        """
        dx = u[0]  # x_dot = u_x
        dy = u[1]  # y_dot = u_y
        return ca.vertcat(dx, dy)

    def solve(
        self,
        x_0: np.ndarray,
        x_goal: np.ndarray,
        x_pred: list[np.ndarray],
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol: ca.OptiSol | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
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
            sol (ca.OptiSol | None, optional): previous solution for warm starting

        Returns:
            u (np.ndarray): optimal control input trajectory (n_u, K)
            x_pred (np.ndarray): predicted state trajectory of the ego agent (n_x, K+1)
            cost (float): optimal cost
            solve_time (float): time taken to solve the OCP in seconds (CPU time)

        Raises:
            RuntimeError: if the OCP has not been initialized yet.
            ValueError: if the input arguments have incorrect shapes.
            ValueError: if use_warm_start is True but sol is not provided.
            RuntimeError: if the OCP solver fails to find a solution.
        """
        # Check if the OCP has been initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "The OCP has not been initialized yet. Please call the "
                "initialize_ocp() method before calling solve()."
            )

        # Validate inputs
        if x_0.shape != (self.n_x,):
            raise ValueError(f"x_0 must have shape ({self.n_x},), but got {x_0.shape}.")
        if x_goal.shape != (self.n_x,):
            raise ValueError(
                f"x_goal must have shape ({self.n_x},), but got {x_goal.shape}."
            )
        if len(x_pred) != self.m - 1:
            raise ValueError(
                f"x_pred must be a list of length {self.m - 1}, but got {len(x_pred)}."
            )
        for j, x_pred_j in enumerate(x_pred):
            if x_pred_j.shape != (self.n_x, self.K + 1):
                raise ValueError(
                    f"x_pred[{j}] must have shape ({self.n_x}, {self.K + 1}), "
                    f"got {x_pred_j.shape} instead."
                )
        if w_curr.shape != (self.m - 1,):
            raise ValueError(
                f"w_curr must have shape ({self.m - 1},), but got {w_curr.shape}."
            )
        if w_target.shape != (self.m - 1,):
            raise ValueError(
                f"w_target must have shape ({self.m - 1},), but got {w_target.shape}."
            )
        if sol is not None and not isinstance(sol, ca.OptiSol):
            raise ValueError(
                f"sol must be of type casadi.OptiSol or None, but got {type(sol)}."
            )
        if sol is not None and sol is None:
            raise ValueError("use_warm_start is True but sol was not provided.")

        # Set parameters in OCP
        self.ocp.set_value(self.x_0, x_0)
        self.ocp.set_value(self.x_goal, x_goal)
        self.ocp.set_value(self.w_curr, w_curr)
        self.ocp.set_value(self.w_target, w_target)
        for j in range(self.m - 1):
            self.ocp.set_value(self.x_pred[j], x_pred[j])

        # Warm start
        if use_warm_start is True and self.sol is not None:
            self.ocp.set_initial(self.sol.value_variables())
        elif self.sol is None:
            # warm start on 1st solution to avoid numerical errors
            # See: https://github.com/casadi/casadi/discussions/3539
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-"NaN-detected"in-my-optimization%3F  # pylint: disable=line-too-long
            for k in range(self.K + 1):
                self.ocp.set_initial(self.x[:, k], x_0)
        else:
            # No warm start
            # CHECKME: check if this case leads to issues
            pass

        # Solve OCP
        self.sol = self.ocp.solve()
        if self.sol.stats()["success"] is not True:
            raise RuntimeError("MPC optimization failed: " + self.sol.stats()["status"])

        # Extract the OCP solution
        u: np.ndarray = self.sol.value(self.u)
        x: np.ndarray = self.sol.value(self.x)
        c: float = self.sol.value(self.cost_function)
        t: float = self.sol.stats()[
            "t_proc_total"
        ]  # CPU time. Alternative: t_wall_total for wall time

        # Return the solution
        return u, x, c, t

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

        # Extract the OCP parameters and solution values
        x = self.sol.value(self.x)
        x_goal = self.sol.value(self.x_goal)
        x_pred = [self.sol.value(x_pred_j) for x_pred_j in self.x_pred]
        w_curr = self.sol.value(self.w_curr)
        w_target = self.sol.value(self.w_target)
        u = self.sol.value(self.u)

        # Goal tracking cost
        goal_cost = 0
        if self.alpha_goal is not None and self.alpha_goal > 0:
            for k in range(self.K + 1):
                goal_cost += self.alpha_goal * (
                    (x[0, k] - x_goal[0]) ** 2 + (x[1, k] - x_goal[1]) ** 2
                )

        # Control input cost
        control_cost = 0
        if self.alpha_u is not None and self.alpha_u > 0:
            for k in range(self.K):
                control_cost += self.alpha_u * (u[:, k].T @ self.R @ u[:, k])

        # Winding cost
        winding_cost = 0
        if self.alpha_w is not None and np.any(self.alpha_w > 0):
            for j in range(self.m - 1):

                # Get weight for agent j
                if isinstance(self.alpha_w, np.ndarray):
                    self.alpha_w: np.ndarray  # to avoid unsubscriptable-object
                    alpha_w_j = self.alpha_w[j]
                else:
                    alpha_w_j = self.alpha_w

                # Compute winding number w.r.t. j at the end of prediction horizon
                w: float = w_curr[j]
                for k in range(1, self.K + 1):
                    theta: float = np.atan2(
                        x[1, k] - x_pred[j][1, k],
                        x[0, k] - x_pred[j][0, k],
                    )
                    theta_prev: float = np.atan2(
                        x[1, k - 1] - x_pred[j][1, k - 1],
                        x[0, k - 1] - x_pred[j][0, k - 1],
                    )
                    w += 1 / (2 * np.pi) * invariants.angle_diff(theta, theta_prev)

                # Cumulate winding costs
                winding_cost += alpha_w_j * (w_target[j] - w) ** 2

        # Total cost
        cost = goal_cost + control_cost + winding_cost

        # Check if the sum of the cost components matches the total cost
        if not np.isclose(cost, self.sol.value(self.cost_function), atol=1e-4):
            raise ValueError(
                f"Cost check failed: the sum of the cost components ({cost}) does not "
                "match the total cost value from the solution "
                f"({self.sol.value(self.cost_function)})."
            )

        return cost, goal_cost, control_cost, winding_cost
