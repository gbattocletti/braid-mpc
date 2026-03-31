from abc import ABC, abstractmethod

import casadi as ca
import numpy as np


class MPC(ABC):

    def __init__(self) -> None:
        """
        Initialize the MPC controller.

        Args:
            None

        Returns:
            None
        """
        # MPC parameters
        self.dt: float | None = None  # time step
        self.K: int | None = None  # prediction horizon
        self.m: int | None = None  # number of agents
        self.n_u: int = 2  # number of control inputs
        self.n_x: int = 2  # number of states

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
            "max_wall_time": 100.0,  # [s]
            "max_cpu_time": 100.0,  # [s]
            "print_level": 0,  # 0-5 (0 = silent, 5 = verbose)
            "tol": 1e-6,  # optimality tolerance
            "acceptable_tol": 1e-4,  # tolerance for early termination
            "linear_solver": "mumps",  # recommended for most problems
        }
        self.sol: ca.OptiSol | None = None  # solution of the OCP

        # Optimization variables
        self.u: ca.Opti.variable  # control input
        self.x: ca.Opti.variable  # state trajectory
        self.cost_function: ca.Opti.variable  # cost function

        # Optimization problem parameters
        self.x_0: ca.Opti.parameter  # initial state
        self.x_goal: ca.Opti.parameter  # goal state
        self.w_curr: ca.Opti.parameter  # current winding number w.r.t. agents
        self.w_target: ca.Opti.parameter  # target winding number w.r.t. agents

        # Cost function weights and matrices
        self.alpha_u: float | None = None  # control input cost weight
        self.alpha_g: float | None = None  # goal tracking cost weight
        self.alpha_g_progress: float | None = None  # weight for progress toward goal
        self.alpha_w: np.ndarray | float | None = None  # winding cost weight
        self.R: np.ndarray | None = None  # control input cost matrix

        # Constraints
        self.u_min: np.ndarray | None = None
        self.u_max: np.ndarray | None = None
        self.u_rate_min: np.ndarray | None = None
        self.u_rate_max: np.ndarray | None = None
        self.u_tot_max: float | None = None  # maximum total control input
        self.x_min: np.ndarray | None = None
        self.x_max: np.ndarray | None = None
        self.d_min: float | None = None  # minimum distance between agents

    def __call__(
        self,
        x_0: np.ndarray,
        x_goal: np.ndarray,
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol_prev: ca.OptiSol | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Call the MPC controller to solve the NMPC problem. See the `solve` method for
        details on the arguments and return values.
        """
        u_opt, x_opt, cost, t_sol = self.solve(
            x_0,
            x_goal,
            w_curr,
            w_target,
            use_warm_start,
            sol_prev,
            **kwargs,
        )
        return u_opt, x_opt, cost, t_sol

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

        # Call the method implemented in the subclass to set up the OCP
        self._initialize_ocp()

    @abstractmethod
    def _initialize_ocp(self):
        raise NotImplementedError(
            "The _initialize_ocp() method must be implemented in a subclass of MPC."
        )

    def dxdt(
        self,
        u: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Continuous-time dynamics function for the MPC controller to be integrated
        numerically within the MPC prediction model. The dynamics are defined as:
            dxdt = f(x, u)

        The state is structured as:
            x = [x, y]^T

        The control input is structured as:
            u = [u_x, u_y]^T

        Args:
            u (ca.SX or ca.MX): Control input vector (n_u, ).

        Returns:
            dxdt (ca.SX or ca.MX): Time derivative of the state vector (n_x, ).
        """
        # Validate input dimensions
        if u.shape not in [(self.n_u,), (1, self.n_u)]:
            raise ValueError(
                f"u must have shape ({self.n_u},) or (1, {self.n_u}), "
                f"but got {u.shape}."
            )

        # Compute the state derivative based on the control input.
        dx = u[0]  # x_dot = u_x
        dy = u[1]  # y_dot = u_y
        dxdt = ca.horzcat(dx, dy)

        # Return the state derivative
        return dxdt

    def solve(
        self,
        x_0: np.ndarray,
        x_goal: np.ndarray,
        w_curr: np.ndarray,
        w_target: np.ndarray,
        use_warm_start: bool = True,
        sol_prev: ca.OptiSol | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Solve the NMPC problem.

        Args:
            x_0 (np.ndarray): initial state of the ego agent (n_x, )
            x_goal (np.ndarray): goal state of the ego agent (n_x, )
            w_curr (np.ndarray): current winding number w.r.t. agents (m-1, )
            w_target (np.ndarray): target winding number w.r.t. agents (m-1, )
            use_warm_start (bool, optional): whether to use the previous solution for
                warm starting
            sol_prev (ca.OptiSol | None, optional): previous solution for warm starting

        Returns:
            u_opt (np.ndarray): optimal control input trajectory (K, n_u)
            x_opt (np.ndarray): predicted state trajectory of the ego agent (K+1, n_x)
            cost (float): optimal cost
            t_sol (float): time taken to solve the OCP in seconds (CPU time)

        Raises:
            RuntimeError: if the OCP has not been initialized yet.
            ValueError: if the input arguments have incorrect shapes.
            ValueError: if use_warm_start is True but sol_prev is not provided.
            RuntimeError: if the OCP solver fails to find a solution.
        """
        # Check if the OCP has been initialized
        if not self.ocp_ready:
            raise RuntimeError(
                "The OCP has not been initialized yet. Please call the "
                "initialize_ocp() method before calling solve()."
            )

        # Call the solve method implemented in the subclass
        self.sol = self._solve(
            x_0,
            x_goal,
            w_curr,
            w_target,
            use_warm_start,
            sol_prev,
            **kwargs,
        )

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

    @abstractmethod
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
        raise NotImplementedError(
            "The _solve() method must be implemented in a subclass of MPC."
        )

    @abstractmethod
    def _check_cost(self) -> float:
        """
        Compute the cost function value for the current solution. This is a helper
        method intended for debugging and testing.

        Args:
            None

        Returns:
            float: optimal cost

        Raises:
            RuntimeError: if the OCP has not been solved yet.
        """
        raise NotImplementedError(
            "The _check_cost() method must be implemented in a subclass of MPC."
        )
