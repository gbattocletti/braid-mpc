import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from braid_controller.core import agent, mpc_centralized
from braid_controller.utils import geometry, invariants, io, weights
from braid_controller.visualization import plot
from braid_controller.visualization.colors import CmdColors

## Compute stats #######################################################################

t_sol_avg = np.mean(t_sol_mat, axis=0)
t_sol_std = np.std(t_sol_mat, axis=0)
t_sol_max = np.max(t_sol_mat, axis=0)
print(
    f"\nAverage solution time: {t_sol_avg[0]:.2f}s "
    f"(std: {t_sol_std[0]:.2f}s, max: {t_sol_max[0]:.2f}s)"
)

## Evaluate results and show plots #####################################################

# Generate plots
plot.plot_paths_3d(
    trajectories[:, :2, :],
    time=time,
    x_lims=np.array(data["x_lims"]),
    y_lims=np.array(data["y_lims"]),
    normalize=False,
    show=False,
)
plot.plot_cost(cost_mat, time, show=False)
plot.plot_windings(windings_target, range(windings_target.shape[0]), show=False)
plot.plot_windings(w_curr_mat, time, windings_ref=w_target_resampled, show=False)
plot.plot_tau(tau_mat, time, tau_i=tau_i_mat, show=False)
# for centralized use: plot.plot_tau(tau_mat, time, show=False)

# Save plots
# TODO
