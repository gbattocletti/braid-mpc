import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from braid_controller.core import agent, mpc_centralized, mpc_distributed
from braid_controller.utils import geometry, invariants, io, weights
from braid_controller.visualization import plot
from braid_controller.visualization.colors import CmdColors

## Settings ############################################################################

# User-defined settings
experiments = [
    "grids_m3_spacelab_t6.yaml",
]

controllers = [
    "distributed",
    "centralized",
    "grid",
]

# Simulation and controller's properties
dt: float = 0.25  # time step
K: int = 21  # MPC horizon
t_end: float = 30  # total simulation time (s)

# Cost function weights
d_min: float = 0.7  # minimum distance between agents
alpha_u: float = 0.05  # control weight
alpha_g: float = 1  # scaling factor for goal tracking cost
alpha_w: float = 20  # scaling factor for winding cost
coeff_s: float = 20  # sharpness of sigmoid function for time-varying weights
coeff_c: float = 0.9  # center of sigmoid function for time-varying weights [0,1]
u_max = np.array([0.5, 0.5])  # maximum control input (m/s)
v_max = np.linalg.norm(u_max)  # maximum speed (m/s)
u_rate_max = np.array([-0.15, -0.15]) * dt  # max velocity change in dt (a * dt)

## Preprocessing #######################################################################
np.random.seed(1312)  # Set random seed for reproducibility
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Move to script directory

for experiment in experiments:

    # Load input data
    filename: str = "data/" + experiment
    data: dict = io.load_yaml(filename)

    # Extract data from yaml file
    m: int = data["m"]
    M = [agent.Agent(i) for i in range(m)]  # create agents objects

    # Check initial and final positions
    x_init: np.ndarray = np.array(data["x_init"]).T
    x_goal: np.ndarray = np.array(data["x_goal"]).T
    if not geometry.check_positions(
        x_init,
        d_min=d_min,
        dynamics="single_integrator",
        u_max=u_max,
        dt=dt,
        verbose=False,
    ):
        raise ValueError(
            "Initial positions are not admissible for collision avoidance."
        )
    if not geometry.check_positions(
        x_goal,
        d_min=d_min,
        dynamics="single_integrator",
        u_max=u_max,
        dt=dt,
        verbose=False,
    ):
        raise ValueError("Goal positions are not admissible for collision avoidance.")

    # Compute target windings from braid or grids
    braid: np.ndarray | None = None
    grids: np.ndarray | None = None
    n_generators: int | None = None
    n_generators_braid: int | None = None
    n_generators_grid: int | None = None
    w_target: np.ndarray | None = None
    n_windings: int | None = None
    if "braid" in data:
        braid = np.array(data["braid"])
        n_generators_braid = braid.shape[0]  # number of generators in the braid word
        n_generators = n_generators_braid
        pass  # TODO compute winding_targets from braid
    if "grids" in data:
        grids = np.array(data["grids"])
        n_generators_grid = grids.shape[0]  # number of generators in the specification
        if w_target is None:
            paths = invariants.grids2paths(grids)
            plot_grids_3d, _ = plot.plot_paths_3d(paths, normalize=True, show=False)
            w_target = invariants.paths2windings(
                paths,
                upscale_factor=1000,
                intermediate_shape="linear",
            )
            n_windings = w_target.shape[0]
            n_generators = n_generators_grid

    # Compute max change in progress variable
    delta_tau: float = 2 / (n_generators * np.pi) * np.arcsin(v_max * dt / d_min)
    delta_tau_K: float = K * delta_tau  # max delta_tau over one MPC horizon

    # Iterate over controllers and run simulations
    for controller in controllers:

        # Reset agents after previous simulation
        for i in range(m):
            M[i].dt = dt
            M[i].x = x_init[:, i]
            M[i].v = np.zeros(3)
            M[i].x_goal = x_goal[:, i]
            M[i].x_opt = np.tile(M[i].x, (K + 1, 1))  # initialize with constant state
            M[i].u_opt = np.zeros((K, 2))  # initialize with zero control inputs
            M[i].t_sol = np.inf
            M[i].cost = np.inf
            M[i].cost_g = np.inf
            M[i].cost_u = np.inf
            M[i].cost_w = np.inf

        # Reset time
        time: np.ndarray = np.arange(0, t_end + dt, dt)

        # Reset MPC parameters (both centralized and distributed)
        alpha_g_k = alpha_g
        alpha_w_k = alpha_w
        theta: np.ndarray = np.zeros((m, m))
        theta_prev: np.ndarray = invariants.relative_headings(x_init)
        w_curr: np.ndarray = np.zeros((m, m))

        # Run MPC controllers
        if controller == "centralized":
            mpc = mpc_centralized.CentralizedMPC(dynamics="single_integrator")
            mpc.dt = dt
            mpc.K = K
            mpc.m = m
            mpc.use_u_rate_constraints = True
            mpc.alpha_u = alpha_u
            mpc.w_epsilon = None
            mpc.d_min = d_min
            mpc.x_min = np.array([data["x_lims"][0], data["y_lims"][0]])
            mpc.x_max = np.array([data["x_lims"][1], data["y_lims"][1]])
            mpc.u_min = -u_max
            mpc.u_max = u_max
            mpc.u_rate_min = -u_rate_max
            mpc.u_rate_max = u_rate_max
            mpc.R = np.diag([1, 1])
            mpc.initialize_ocp()

            x_pred = None
            x_prev = None

            # Run control loop
            for step, t in enumerate(time):
                pass

        elif controller == "distributed":
            mpc = mpc_distributed.DistributedMPC(dynamics="single_integrator")
            mpc.dt = dt
            mpc.K = K
            mpc.m = m
            mpc.collision_avoidance = "convex"
            mpc.slack_collision_constraints = False
            mpc.slack_state_constraints = False
            mpc.use_u_rate_constraints = True
            mpc.alpha_u = alpha_u
            mpc.w_epsilon = None
            mpc.d_min = d_min
            mpc.x_min = np.array([data["x_lims"][0], data["y_lims"][0]])
            mpc.x_max = np.array([data["x_lims"][1], data["y_lims"][1]])
            mpc.u_min = -u_max
            mpc.u_max = u_max
            mpc.u_rate_min = -u_rate_max
            mpc.u_rate_max = u_rate_max
            mpc.R = np.diag([1, 1])
            mpc.initialize_ocp()

            # Initialize horizon helper variable
            horizon: np.ndarray = np.arange(0, K + 1)  # {0, ..., K}, K+1 elements

            # Initialize helper variables to store trajectory predictions
            x_pred = np.zeros([K + 1, mpc.n_x, m])  # shape: (K+1, n_x, m)
            x_prev = np.zeros([K + 1, mpc.n_x])  # shape: (K+1, n_x).

            # Run control loop
            for step, t in enumerate(time):
                pass

        elif controller == "grid":
            continue  # TODO: implement grid controller
