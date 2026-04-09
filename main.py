# pylint: disable=unused-import
from typing import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rps.robotarium as rb

from core import agent, mpc_centralized, mpc_distributed
from data import grids_1, grids_2
from utils import invariants, robotarium_bridge
from visualization import plot

## Settings ############################################################################
DATA = grids_2  # initial and goal locations, topological specification
CONTROL_ARCHITECTURE = "centralized"  # "distributed" or "centralized"
USE_ROBOTARIUM = False  # otherwise, dynamics from the agents' objects is used
SHOW_PLOTS = True
DEBUG = True

## Preprocessing #######################################################################
np.random.seed(1312)

# Load input data
m = DATA.m
grids = DATA.grids  # topological specification
x_init = DATA.x_init.T
x_goal = DATA.x_goal.T

# Add orientation dimension to states if not included
# NOTE: heading in x_init is used only for unicycle dynamic model, while goal heading
# is ignored in cost function and is added only for dimension consistency.
x_init = np.vstack([x_init, np.zeros([1, m])]) if x_init.shape[0] == 2 else x_init
x_goal = np.vstack([x_goal, np.zeros([1, m])]) if x_goal.shape[0] == 2 else x_goal

# Convert grids to paths
paths = invariants.grids2paths(grids)
plot.plot_paths_3d(paths, show=False)

# Compute winding numbers
windings = invariants.paths2windings(
    paths,
    upscale_factor=100,
    intermediate_shape="linear",
)
n_windings = windings.shape[0]  # length of the winding number vector
plot.plot_windings(windings, show=False)

## Controllers and agents setup ########################################################
# Initialize controller's properties
dt = 0.1  # s
K = 20  # time steps

# Create agents
M = [agent.Agent(i) for i in range(m)]
for i in range(m):
    M[i].dt = dt
    M[i].x = x_init[:, i]
    M[i].x_goal = x_goal[:, i]

# Initialize MPC controller
dynamics = "unicycle" if USE_ROBOTARIUM is True else "single_integrator"
if CONTROL_ARCHITECTURE == "distributed":
    mpc = mpc_distributed.DistributedMPC(dynamics=dynamics)
elif CONTROL_ARCHITECTURE == "centralized":
    mpc = mpc_centralized.CentralizedMPC(dynamics=dynamics)
else:
    raise ValueError(f"Invalid control architecture: {CONTROL_ARCHITECTURE}")
mpc.dt = dt
mpc.K = K
mpc.m = m
mpc.alpha_u = 0.000001
mpc.alpha_g = 0.000001
mpc.alpha_w = 1000
mpc.R = np.diag([1, 1])
# NOTE: when using the robotarium, the velocity limits (u limits) must currently be
# scaled manually to match the robotarium's velocity limits, depending on the scaling
# factor used between the robotarium environment size and the 'real world' environment
# size. Otherwise, the robot's velocity will be different than expected.
mpc.u_min = np.array([-0.625, -1.25])
mpc.u_max = np.array([0.625, 1.25])
mpc.x_min = np.array([0, 0])
mpc.x_max = np.array([10, 10])
mpc.d_min = 0.05
mpc.initialize_ocp()

# Initialize helper variables (only for distributed MPC)
x_pred = np.zeros([K, mpc.n_x, m]) if mpc.architecture == "distributed" else None

## Simulation setup ####################################################################
if USE_ROBOTARIUM is True:
    # Initialize robotarium
    x_init_robotarium: np.ndarray = robotarium_bridge.real2robotarium(
        x_init,
        [mpc.x_min[0][0], mpc.x_max[0][0]],
        [mpc.x_min[0][1], mpc.x_max[0][1]],
    )
    x_goal_robotarium = robotarium_bridge.real2robotarium(
        x_goal,
        [mpc.x_min[0][0], mpc.x_max[0][0]],
        [mpc.x_min[0][1], mpc.x_max[0][1]],
    )

    # Initialize robotarium
    r = rb.Robotarium(
        number_of_robots=m,
        initial_conditions=x_init_robotarium,
        show_figure=True,
        sim_in_real_time=True,
    )
    x_meas = r.get_poses()  # get states in robotarium

    # Initialize velocity vector for robotarium control inputs
    v_vec = np.zeros((mpc.n_u, m))

    # Plot relevant information
    colors = plt.color_sequences["tab10"][:m]
    rect = patches.Rectangle(
        (-1.6, -1), 2, 2, linewidth=2, edgecolor="black", facecolor="none"
    )
    r.axes.add_patch(rect)  # plot environment boundary
    r.axes.set_xlim(-1.6, 0.4)  # override robotarium default limits
    r.axes.set_ylim(-1, 1)
    r.axes.set_aspect("equal", adjustable="box")
    r.axes.grid(
        True,
        which="major",
        linestyle=":",
        color="gray",
        linewidth=0.5,
        zorder=1,
    )
    r.axes.grid(
        True,
        which="minor",
        linestyle=":",
        color="gray",
        linewidth=0.3,
        zorder=1,
    )
    for i in range(m):
        r.axes.plot(
            x_init_robotarium[0, i],
            x_init_robotarium[1, i],
            "o",
            color=colors[i],
        )  # plot initial position
        r.axes.plot(
            x_goal_robotarium[0, i],
            x_goal_robotarium[1, i],
            "*",
            color=colors[i],
        )  # plot goal position

## Main simulation loop ################################################################

# Initialize time vector
T: float = 10  # total simulation time (s)
time: np.ndarray = np.arange(0, T + dt, dt)

# Initialize matrix to store traveled trajectories
traj: np.ndarray = np.zeros((len(time), mpc.n_x, m))

for step, t in enumerate(time):

    # 1. Compute current and target winding numbers
    # NOTE: currently the time progresses constantly along the braid
    tau = t / T  # time n
    w_curr = windings[int(tau * (n_windings - 1)), :, :]
    tau_target = min((t + K * dt) / T, 1)  # cap target time at the end of the braid
    w_target = windings[int(tau_target * (n_windings - 1)), :, :]

    # 2. Solve MPC problem
    if mpc.architecture == "distributed":
        # Collect predicted trajectories of all agents
        for i in range(m):
            x_pred[0:, :, i] = M[i].x
            if M[i].x_opt is None:
                x_pred[1:, :, i] = M[i].x  # use constant state as predicted trajectory
            else:
                x_pred[1:, :, i] = M[i].x_opt[2:]

        # Solve local MPC problem for each agent
        for i in range(m):
            (u_opt, x_opt, cost, t_sol) = mpc.solve(
                x_0=M[i].x,
                x_goal=M[i].x_goal,
                w_curr=np.delete(w_curr[i], i, axis=0),  # remove self winding number
                w_target=np.delete(w_target[i], i, axis=0),
                use_warm_start=True,
                sol=M[i].sol,
                x_pred=np.delete(x_pred, i, axis=2),  # remove self predicted trajectory
            )
            M[i].sol = mpc.sol  # save solution in agent object
            M[i].u_opt = u_opt
            M[i].x_opt = x_opt
            M[i].cost = cost
            M[i].t_sol = t_sol

    # Centralized MPC controller
    elif mpc.architecture == "centralized":
        # Collect initial states of all agents
        x_0 = [M[i].x for i in range(m)]
        x_0 = np.array(x_0).T  # reshape to m x n_x

        # Solve centralized MPC problem
        (u_opt, x_opt, cost, t_sol) = mpc.solve(
            x_0=x_0,
            x_goal=x_goal,
            w_curr=w_curr,
            w_target=w_target,
            use_warm_start=True,
            sol_prev=mpc.sol,
        )

        # Save solution in agent objects
        for i in range(m):
            M[i].u_opt = u_opt[i]
            M[i].x_opt = x_opt[i]
            M[i].cost = cost  # same cost for all agents in centralized MPC
            M[i].t_sol = t_sol  # same solution time for all agents in centralized MPC

    # 3. Take one step in the environment and get new state
    if USE_ROBOTARIUM is True:

        # Convert optimal control inputs to robotarium velocities
        for i in range(m):
            v_vec[:, i] = robotarium_bridge.real2robotarium_vel(
                M[i].u_opt[0],
                mpc.u_min,
                mpc.u_max,
            )

        # Simulate with Robotarium
        # NOTE: The robotarium uses a fixed time step of 0.033s, so we need to call the
        # step function multiple times to match the desired MPC time step (dt).
        for _ in range(int(dt / r.time_step)):
            r.set_velocities(range(m), v_vec)
            r.step()
            x_meas = r.get_poses()  # get new states (required after each r.step())

        # Update agent's states
        for i in range(m):
            M[i].x = robotarium_bridge.robotarium2real(
                x_meas[:, i],
                [mpc.x_min[0][0], mpc.x_max[0][0]],
                [mpc.x_min[0][1], mpc.x_max[0][1]],
            )

    else:

        # Simulate step using agent's dynamics
        for i in range(m):
            M[i].step(M[i].u_opt[0])

    # 4. Save traveled trajectory
    for i in range(m):
        traj[step, :, i] = M[i].x

    # 5. Print debug info
    if DEBUG is True:
        print(f"t: {t:5.2f}s")
        for i in range(m):
            print(
                f"\ti: {i}, "
                f"x(k): [{M[i].x_opt[0, 0]:5.2f},{M[i].x_opt[0, 1]:5.2f},"
                f"{M[i].x_opt[0, 2]:5.2f}], "
                f"x(k+1): [{M[i].x[0]:5.2f},{M[i].x[1]:5.2f},{M[i].x[2]:5.2f}], "
                f"u*(0|k): [{M[i].u_opt[0, 0]: 4.2f},{M[i].u_opt[0, 1]: 4.2f}], "
                f"t_sol: {M[i].t_sol:.2f}s, "
                f"cost: {M[i].cost:.2f}"
            )

# Call robotarium termination function
if USE_ROBOTARIUM is True:
    r.call_at_scripts_end()

## Evaluate results and show plots #####################################################
plot.plot_paths_3d(traj[:, :2, :], show=False)

# Show all plots
if SHOW_PLOTS is True:
    plt.show()
