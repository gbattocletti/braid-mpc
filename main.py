import matplotlib.pyplot as plt
import numpy as np
import rps.robotarium as rb

from core import agent, mpc
from data import grids_1, grids_2  # pylint: disable=unused-import
from utils import invariants, robotarium_bridge
from visualization import plot

## Settings ############################################################################
DEBUG = True
USE_ROBOTARIUM = True
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

## Controller and agents setup #########################################################
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
# NOTE: when using the robotarium, the velocity limits (u limits) must currently be
# scaled manually to match the robotarium's velocity limits, depending on the scaling
# factor used between the robotarium environment size and the 'real world' environment
# size. Otherwise, the robot's velocity will be different than expected.
mpc_distributed = mpc.MPC()
mpc_distributed.dt = dt
mpc_distributed.K = K
mpc_distributed.m = m
mpc_distributed.alpha_u = 0.001
mpc_distributed.alpha_goal = 1000.0
mpc_distributed.alpha_w = 0
mpc_distributed.R = np.diag([1, 1])
mpc_distributed.u_min = np.array([-0.625, -0.625])
mpc_distributed.u_max = np.array([0.625, 0.625])
mpc_distributed.x_min = np.array([0, 0])
mpc_distributed.x_max = np.array([10, 10])
mpc_distributed.d_min = 0.01  # slightly larger than needed
mpc_distributed.initialize_ocp()

# Initialize helper variables
x_pred = np.zeros([K, 2, m])
x_pred_new = np.zeros([K, 2, m])
v_vec = np.zeros((2, m))  # only used for robotarium

## Simulation setup ####################################################################
if USE_ROBOTARIUM is True:
    x_init_robotarium: np.ndarray = np.vstack(
        [
            robotarium_bridge.real2robotarium(
                x_init,
                [mpc_distributed.x_min[0][0], mpc_distributed.x_max[0][0]],
                [mpc_distributed.x_min[0][1], mpc_distributed.x_max[0][1]],
                coords_type="position",
            ).T,
            np.zeros([1, m]),
        ]
    )
    r = rb.Robotarium(
        number_of_robots=m,
        initial_conditions=x_init_robotarium,
        show_figure=True,
        sim_in_real_time=True,
    )
    x_meas = r.get_poses()

## Main simulation loop ################################################################
T = 20  # total simulation time (s)
time = np.arange(0, T + dt, dt)
for t in time:

    # Compute current and target winding numbers
    # NOTE: currently the time progresses constantly along the braid
    tau = t / T  # time n
    w_curr = windings[int(tau * (n_windings - 1)), :, :]
    tau_target = min((t + K * dt) / T, 1)  # cap target time at the end of the braid
    w_target = windings[int(tau_target * (n_windings - 1)), :, :]

    # Iterate over agents and solve their local MPC problems and apply control inputs
    for i in range(m):
        ego_agent = M[i]
        x_start = ego_agent.x  # for debugging only

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
        if USE_ROBOTARIUM is True:
            v_vec[:, i] = robotarium_bridge.real2robotarium(
                u_opt[0],
                [mpc_distributed.u_min[0][0], mpc_distributed.u_max[0][0]],
                [mpc_distributed.u_min[0][1], mpc_distributed.u_max[0][1]],
                coords_type="velocity",
            )  # set velocity command for agent i in robotarium
        ego_agent.step(u_opt[0])  # simulate step using agent's dynamics
        x_pred_new[0, :, i] = ego_agent.x  # save 'true' new state to predicted traj
        x_pred_new[1:, :, i] = x_opt[2:, :]  # store predicted traj for next step
        ego_agent.sol = mpc_distributed.sol  # save solution for warm start @ next step

        # Print debug info
        if DEBUG is True:
            print(
                f"t: {t:5.2f}s, i: {i}, x(k): [{x_start[0]:5.2f},{x_start[1]:5.2f}], "
                f"x(k+1): [{ego_agent.x[0]:5.2f},{ego_agent.x[1]:5.2f}], "
                f"u*(0|k): [{u_opt[0][0]: 4.2f},{u_opt[0][1]: 4.2f}], "
                f"t_sol: {t_sol:.2f}s, cost: {cost:.2f}"
            )

    # Simulate with Robotarium
    if USE_ROBOTARIUM is True:

        # Execute simulation step
        r.set_velocities(range(m), v_vec)
        r.step()

        # Overwrite 'true' new states with the ones measured from Robotarium
        x_meas = r.get_poses()
        for i in range(m):
            x_pred_new[0, :, i] = robotarium_bridge.robotarium2real(
                x_meas[:, i],
                [mpc_distributed.x_min[0][0], mpc_distributed.x_max[0][0]],
                [mpc_distributed.x_min[0][1], mpc_distributed.x_max[0][1]],
                coords_type="position",
            )[0:-1]
            M[i].x = x_pred_new[0, :, i]

    # Update variables for the next iteration
    x_pred = x_pred_new

## Evaluate results and show plots #####################################################


# Show all plots
plt.show()
