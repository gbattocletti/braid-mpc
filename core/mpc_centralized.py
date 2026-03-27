import casadi as ca
import numpy as np

from core.mpc import MPC


class CentralizedMPC(MPC):
    """
    Centralized MPC controller class. The main structure of the class and of the OCP
    formulation is the same as for the base MPC class, but the OCP is formulated in a
    centralized way, meaning that the cost, dynamics, and constraints, of all the agents
    are included in a single optimization problem. This also means that, differently
    from the local MPC controller, the variables and parameters of this class contain
    the data for all the agents, and thus tehy have an extra dimension of length m
    (number of agents).

    Additionally, the centralized OCP does not need the predicted trajectories of the
    other agents, as all the trajectories are optimized simultaneously.
    """

    def __init__(self) -> None:
        super().__init__()
        self.solver_options["max_wall_time"] = 600.0  # [s]
        self.solver_options["max_cpu_time"] = 600.0  # [s]

    def _initialize_ocp(self) -> None:
        """
        Initialize the centralized optimal control problem (OCP) for the centralized MPC
        controller. This method must be called before calling the `solve` method for the
        first time.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: if the OCP has already been initialized.
        """

        # Initialize optimization variables
        self.x = self.ocp.variable(self.n_x, self.K + 1, self.m)
        self.u = self.ocp.variable(self.n_u, self.K, self.m)

        # Initialize OCP parameters
        self.x_0 = self.ocp.parameter(self.m, self.n_x)
        self.x_goal = self.ocp.parameter(self.m, self.n_x)
        self.w_curr = self.ocp.parameter(self.m, self.m)
        self.w_target = self.ocp.parameter(self.m, self.m)

        # Constraints
        # Initial state constraint
        self.ocp.subject_to(self.x[:, 0, :] == self.x_0)  # CHECKME: is reshape needed?

        # TODO

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
