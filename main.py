import matplotlib.pyplot as plt
import numpy as np

from core import agent, mpc
from data import grids_1, grids_2  # pylint: disable=unused-import
from utils import invariants
from visualization import plot

# Settings
DEBUG = True
data = grids_2
np.random.seed(1312)

## Preprocessing #######################################################################
# Load input data
m = data.m
grids = data.grids  # topological specification
x_init = data.x_init
x_goal = data.x_goal

# Convert grids to paths
paths = invariants.grids2paths(grids)
plot.plot_paths_3d(paths, show=False)

# Compute winding numbers
windings = invariants.paths2windings(
    paths,
    upscale_factor=10,
    intermediate_shape="spline",
)
n_windings = windings.shape[0]  # length of the winding number vector
plot.plot_windings(windings, show=False)

## Controller Setup ##################################################################
# Initialize controller's properties
dt = 0.5  # s
K = 20  # time steps

# Create agents
M = [agent.Agent(i) for i in range(m)]
for i in range(m):
    M[i].dt = dt
    M[i].x = x_init[i, :]
    M[i].x_goal = x_goal[i, :]

# Initialize MPC controller (local MPC for distributed control architecture)
mpc_distributed = mpc.MPC()
mpc_distributed.dt = dt
mpc_distributed.K = K
mpc_distributed.m = m
mpc_distributed.alpha_u = 1.0
mpc_distributed.alpha_goal = 1.0
mpc_distributed.alpha_w = 1.0
mpc_distributed.R = np.diag([1, 1])
mpc_distributed.u_min = np.array([-1, -1])  # TODO: check values against robotarium
mpc_distributed.u_max = np.array([1, 1])
mpc_distributed.x_min = np.array([-10, -10])
mpc_distributed.x_max = np.array([10, 10])
mpc_distributed.d_min = 0.2
mpc_distributed.initialize_ocp()

# Initialize helper variables
x_pred = np.zeros([K, 2, m])
x_pred_new = np.zeros([K, 2, m])

# Main simulation loop
T = 50  # total simulation time (s)
time = np.arange(0, T + dt, dt)
for t in time:

    # Compute current and target winding numbers
    # NOTE: currently the time progresses constantly along the braid
    tau = t / T  # time n
    tau_target = (t + K * dt) / T
    w_curr = windings[int(tau * n_windings), :, :]
    w_target = windings[int(tau_target * n_windings), :, :]

    # Iterate over agents and solve their local MPC problems and apply control inputs
    for i in range(m):
        ego_agent = M[i]

        # Solve local MPC problem for the ego agent
        (u_opt, x_opt, cost, t_sol) = mpc_distributed.solve(
            x_0=ego_agent.x,
            x_goal=ego_agent.x_goal,
            x_pred=np.delete(x_pred, i, axis=2),  # remove ego agent's predicted traj
            w_curr=np.delete(w_curr[i], i, axis=0),  # remove winding number w.r.t. self
            w_target=np.delete(w_target[i], i, axis=0),
            use_warm_start=True,
            sol=ego_agent.sol,
        )

        # Extract solution and apply control input
        ego_agent.step(u_opt[0])
        ego_agent.sol = mpc_distributed.sol
        x_pred_new[0, :, i] = ego_agent.x  # save 'true' new state to predicted traj
        x_pred_new[1:, :, i] = x_opt[2:, :]  # store predicted traj for next step

        # Print debug info
        if DEBUG is True:
            print(f"Time: {t:.2f} s, Agent {i}, Control input: {u_opt[0]}")

    # Update variables for the next iteration
    x_pred = x_pred_new

# Show all plots
plt.show()
