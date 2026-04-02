import casadi as ca
import numpy as np


class Agent:
    def __init__(self, agent_id: int) -> None:
        """
        Initialize the agent.

        Args:
            agent_id (int): Unique identifier for the agent.

        Returns:
            None
        """
        # Agent properties
        self.id: int = agent_id
        self.dt: float | None = None  # time step

        # Agent state info
        self.x: np.ndarray | None = None  # current state of the agent
        self.x_opt: np.ndarray | None = None  # state trajectory from MPC solution
        self.u_opt: np.ndarray | None = None  # control trajectory from MPC solution

        # Agent goal info
        self.x_goal: np.ndarray | None = None  # goal state of the agent

        # MPC solution object (for warm starting the optimization problem)
        self.sol: ca.OptiSol | None = None  # solution of the OCP
        self.cost: float | None = None  # cost of the MPC solution
        self.t_sol: float | None = None  # time taken to solve the MPC problem

    def step(self, u: np.ndarray) -> None:
        """
        Take an action based on the control input and the agent's dynamics.

        Args:
            u (np.ndarray): Control input for the agent.

        Returns:
            None
        """
        # Validate input
        if len(u) == 2:
            u = np.append(u, 0)  # add dummy control input for heading

        # simple dynamics (single integrator): x_next = x_prev + u*dt
        self.x = self.x + u * self.dt
