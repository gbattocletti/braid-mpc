from abc import ABC, abstractmethod
from typing import Callable

import casadi as ca
import numpy as np


class MPC(ABC):

    def __init__(self, dynamics: str = "single_integrator") -> None:
        """
        Initialize the MPC controller.

        Args:
            dynamics: The dynamics model to use ("single_integrator" or "unicycle")

        Returns:
            None
        """
        # Debug level
        self.debug: bool = False

        # Controller architecture
        self.architecture: str = "centralized"  # {"centralized", "distributed"}

        # MPC dynamics
        self.dynamics: str = dynamics  # {"single_integrator", "unicycle"}
        self.dxdt: Callable[[ca.SX | ca.MX, ca.SX | ca.MX], ca.SX | ca.MX]
        if self.dynamics == "single_integrator":
            self.dxdt = self._dxdt_si
        elif self.dynamics == "unicycle":

            self.dxdt = self._dxdt_uni
        else:
            raise ValueError(f"Invalid dynamics: {self.dynamics}")

        # MPC parameters
        self.dt: float | None = None  # time step
        self.K: int | None = None  # prediction horizon
        self.m: int | None = None  # number of agents
        self.n_u: int = 2  # number of control inputs
        self.n_x: int = 3  # number of states
        self.n_x_pos: int = 2  # number of position states

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
        self.w_epsilon: float  # bound on the difference between w and w_target

        # Cost function weights and matrices
        self.alpha_g: ca.Opti.parameter  # goal tracking cost weight (time varying)
        self.alpha_u: float  # control input cost weight (constant)
        self.R: np.ndarray  # control input cost matrix (constant)
        self.alpha_w: ca.Opti.parameter  # winding cost weight (time varying)

        # Constraints
        self.u_min: np.ndarray | None = None
        self.u_max: np.ndarray | None = None
        self.u_rate_min: np.ndarray | None = None
        self.u_rate_max: np.ndarray | None = None
        self.u_tot_max: float | None = None  # maximum total control input
        self.x_min: np.ndarray | None = None  # minimum position state
        self.x_max: np.ndarray | None = None  # maximum position state
        self.d_min: float | None = None  # minimum distance between agents' positions
        self.constraints: list = []  # for debugging purposes

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

    def set_alpha_g(self, alpha_g: float) -> None:
        """
        Set the goal tracking cost weight.

        Args:
            alpha_g (float): goal tracking cost weight

        Returns:
            None
        """
        self.ocp.set_value(self.alpha_g, alpha_g)

    @abstractmethod
    def set_alpha_w(self, alpha_w: np.ndarray) -> None:
        """
        Set the winding cost weight.

        Args:
            alpha_w (np.ndarray): winding cost weight matrix.

        Returns:
            None

        Raises:
            NotImplementedError: if the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The set_alpha_w() method must be implemented in a subclass of MPC, since "
            "the shape of alpha_w depends on the specific MPC architecture."
        )

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

    def _dxdt_si(
        self,
        _: ca.SX | ca.MX,
        u: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Single integrator continuous-time dynamics function to be integrated
        numerically within the MPC prediction model.

        Args:
            _ (ca.SX or ca.MX): State vector (n_x, ). Unused but included for
                consistency with the signature of dxdt functions for different dynamics.
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
        dz = 0  # dummy state derivative for heading, which is not used in si model
        dxdt = ca.horzcat(dx, dy, dz)

        # Return the state derivative
        return dxdt

    def _dxdt_uni(
        self,
        x: ca.SX | ca.MX,
        u: ca.SX | ca.MX,
    ) -> ca.SX | ca.MX:
        """
        Unicycle continuous-time dynamics function to be integrated numerically within
        the MPC prediction model.

        Args:
            x (ca.SX or ca.MX): State vector (n_x, ).
            u (ca.SX or ca.MX): Control input vector (n_u, ).

        Returns:
            dxdt (ca.SX or ca.MX): Time derivative of the state vector (n_x, ).
        """
        # Validate input dimensions
        if x.shape not in [(self.n_x,), (1, self.n_x)]:
            raise ValueError(
                f"x must have shape ({self.n_x},) or (1, {self.n_x}), "
                f"but got {x.shape}."
            )
        if u.shape not in [(self.n_u,), (1, self.n_u)]:
            raise ValueError(
                f"u must have shape ({self.n_u},) or (1, {self.n_u}), "
                f"but got {u.shape}."
            )

        # Compute the state derivative based on the control input.
        dx = u[0] * ca.cos(x[2])  # x_dot = u_x*cos(theta)
        dy = u[0] * ca.sin(x[2])  # y_dot = u_y*sin(theta)
        dtheta = u[1]  # theta_dot = u_theta
        dxdt = ca.horzcat(dx, dy, dtheta)

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
            x_0 (np.ndarray): initial state
            x_goal (np.ndarray): goal state
            w_curr (np.ndarray): current winding numbers
            w_target (np.ndarray): target winding numbers
            use_warm_start (bool, optional): whether to use the previous solution for
                warm starting
            sol_prev (ca.OptiSol | None, optional): previous solution for warm starting

        Returns:
            u_opt (np.ndarray | list[np.ndarray]): optimal control input trajectory
            x_opt (np.ndarray | list[np.ndarray]): optimal state trajectory
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

        # Solve the OCP by calling the _solve method implemented in the subclass
        self.sol = self._solve(
            x_0,
            x_goal,
            w_curr,
            w_target,
            use_warm_start,
            sol_prev,
            **kwargs,
        )

        # Extract the OCP solution
        if self.architecture == "distributed":
            u_opt: np.ndarray = self.sol.value(self.u)
            x_opt: np.ndarray = self.sol.value(self.x)
        elif self.architecture == "centralized":
            u_opt: list[np.ndarray] = [self.sol.value(self.u[i]) for i in range(self.m)]
            x_opt: list[np.ndarray] = [self.sol.value(self.x[i]) for i in range(self.m)]
        else:
            raise ValueError(f"Invalid architecture: {self.architecture}")
        c: float = self.sol.value(self.cost_function)
        t: float = self.sol.stats()[
            "t_proc_total"
        ]  # CPU time. Alternative: t_wall_total for wall time

        # Return the solution
        return u_opt, x_opt, c, t

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
    def check_cost(self) -> tuple[float, float, float, float]:
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
            "The check_cost() method must be implemented in a subclass of MPC."
        )
