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
        self.x: np.ndarray | None = None  # current state of the agent (initial state)
        self.x_goal: np.ndarray | None = None  # goal state of the agent
        self.x_pred: np.ndarray | None = None  # agent's predicted trajectory

        # MPC solution object (for warm starting the optimization problem)
        self.sol: ca.OptiSol | None = None  # solution of the OCP

    def step(self, u: np.ndarray) -> None:
        """
        Take an action based on the control input.

        Args:
            u (np.ndarray): Control input for the agent.

        Returns:
            None
        """
        self.x = self.x + u * self.dt  # simple dynamics: x_next = x_prev + u*dt
