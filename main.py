import matplotlib.pyplot as plt
import numpy as np

from core import agent, mpc
from data import grids_2
from utils import invariants
from visualization import plot

# Load input data
grids = grids_2.grids

# Convert grids to paths
paths = invariants.grids2paths(grids)
plot.plot_paths_3d(paths, show=False)

# Compute winding numbers
windings_spline = invariants.paths2windings(
    paths,
    upscale_factor=20,
    intermediate_shape="spline",
)
plot.plot_windings(windings_spline, show=False)

# Initialize agents' properties
# NOTE: x_init and x_goal must be coherent with initial and final relative positions in
# the topological specifications.
x_init = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
)
x_goal = np.array(
    [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
)

# Initialize controller's properties
dt = 1  # s NOTE: also used by the agents
K = 20  # time steps

# Create agents
m = paths.shape[-1]
M = [agent.Agent(i) for i in range(m)]
for i in range(m):
    M[i].dt = dt
    M[i].x = x_init[i, :]
    M[i].x_init = x_init[i, :]
    M[i].x_goal = x_goal[i, :]
    M[i].x_pred = np.zeros([m, m, K])

# Initialize MPC controller
mpc = mpc.MPC()
mpc.dt = dt
mpc.K = K

# Main simulation loop

# Show all plots
plt.show()
