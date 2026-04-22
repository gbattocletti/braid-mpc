import os

import matplotlib.pyplot as plt
import numpy as np
import rps.robotarium as rb
from matplotlib import patches

from core import agent, mpc_centralized, mpc_distributed
from utils import geometry, invariants, io, robotarium_bridge
from visualization import plot, plot_debug

## Settings ############################################################################

# User-defined settings
DATA = "data/grids_m5_2.yaml"  # initial and goal locations, topological specification
CONTROL_ARCHITECTURE = "distributed"  # "distributed" or "centralized"
USE_ROBOTARIUM = False  # otherwise, dynamics from the agents' objects is used
SHOW_PLOTS = True
DEBUG = True
DEBUG_HYPERPLANES = False  # whether to plot the hyperplanes and stop the simulation

# Select progress strategy along the specification
# Available approaches are:
# - uniform_time --> uniform progression over time
# - winding_progress --> estimate progression from actual trajectories
# In case of distributed control, an additional selection can be made to determine the
#  selection of tau from individual estimates. Can be either "mean" or "min".
PROGRESS_STRATEGY = "winding_progress"
PROGRESS_STRATEGY_DISTRIBUTED = "min"

# Simulation and controller's properties
DT: float = 0.1  # s
K: int = 20  # time steps
T: float = 100  # total simulation time (s)

# Cost function weights
ALPHA_U: float = 0.001  # control cost (constant).
ALPHA_G: float = 0.1  # scaling factor for goal tracking cost; use 0 to disable
EXP_GOAL: int = 20  # exponent for time-varying goal cost weight
ALPHA_W: float = 1e3  # scaling factor for winding cost; use 0 to disable
USE_TIME_VARYING_WEIGHTS: bool = True

## Preprocessing #######################################################################

# Move to script directory
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)

# Load input data
data = io.load_yaml(DATA)
m = data["m"]
grids = np.array(data["grids"])  # topological specification
n_generators = grids.shape[0]  # number of generators in the specification
x_init = np.array(data["x_init"]).T
x_goal = np.array(data["x_goal"]).T

# Add orientation dimension to states if not included
# NOTE: heading in x_init is used only for unicycle dynamic model, while goal heading
# is ignored in cost function and is added only for dimension consistency.
x_init = np.vstack([x_init, np.zeros([1, m])]) if x_init.shape[0] == 2 else x_init
x_goal = np.vstack([x_goal, np.zeros([1, m])]) if x_goal.shape[0] == 2 else x_goal

# Convert grids to paths
paths = invariants.grids2paths(grids)
plot.plot_paths_3d(paths, normalize=True, show=False)

# Compute target winding numbers
windings_target = invariants.paths2windings(
    paths,
    upscale_factor=1000,
    intermediate_shape="linear",
)  # (n_windings, m, m)
n_windings = windings_target.shape[0]  # length of the winding number vector

# Set random seed for reproducibility
np.random.seed(1312)

## Controllers and agents setup ########################################################

# Create agents
M = [agent.Agent(i) for i in range(m)]
for i in range(m):
    M[i].dt = DT
    M[i].x = x_init[:, i]
    M[i].x_goal = x_goal[:, i]
    M[i].x_opt = np.tile(M[i].x, (K + 1, 1))  # initialize with constant state
    M[i].u_opt = np.zeros((K, 2))  # initialize with zero control inputs
    M[i].t_sol = np.inf
    M[i].cost = np.inf
    M[i].cost_g = np.inf
    M[i].cost_u = np.inf
    M[i].cost_w = np.inf

# Initialize MPC controller
dynamics = "unicycle" if USE_ROBOTARIUM is True else "single_integrator"
if CONTROL_ARCHITECTURE == "distributed":
    mpc = mpc_distributed.DistributedMPC(dynamics=dynamics)
elif CONTROL_ARCHITECTURE == "centralized":
    mpc = mpc_centralized.CentralizedMPC(dynamics=dynamics)
else:
    raise ValueError(f"Invalid control architecture: {CONTROL_ARCHITECTURE}")
mpc.dt = DT
mpc.K = K
mpc.m = m
mpc.alpha_u = ALPHA_U  # constant (in general)
mpc.d_min = 2
mpc.x_min = np.array([data["x_lims"][0], data["y_lims"][0]])
mpc.x_max = np.array([data["x_lims"][1], data["y_lims"][1]])
if USE_ROBOTARIUM is True:
    # NOTE: when using the robotarium, the linear velocity limit (1st element of the
    # u_min and u_max vectors) must be scaled to match the robotarium's velocity limits.
    # The scaling depends on the scaling factor between the robotarium environment size
    # and the 'real world' environment size. Otherwise, the robot's velocity will be
    # different than expected.
    mpc.u_min = np.array([-0.625, -1.25])  # u is a vector [v, w]
    mpc.u_max = np.array([0.625, 1.25])
    mpc.R = np.diag([1, 0.1])  # lower penalty for angular velocity
else:
    mpc.u_min = np.array([-1, -1])  # u is a vector [v_x, v_y]
    mpc.u_max = np.array([1, 1])
    mpc.R = np.diag([1, 1])
mpc.initialize_ocp()

# Initialize helper variables to store trajectory predictions (only for distributed MPC)
x_pred = np.zeros([K + 1, mpc.n_x, m]) if mpc.architecture == "distributed" else None
x_prev = np.zeros([K + 1, mpc.n_x]) if mpc.architecture == "distributed" else None

# Initialize max progress speed variable
if PROGRESS_STRATEGY == "winding_progress":
    progress_coefficient = 1  # in [0,1], to slow progress if delta_tau_max is too big
    if mpc.dynamics == "single_integrator":
        v_max = np.linalg.norm(mpc.u_max)
    else:
        v_max = mpc.u_max[0]
    delta_tau_max: float = progress_coefficient * (
        2 * K / (n_generators * np.pi) * np.arcsin(v_max * DT / mpc.d_min)
    )  # max delta_tau over one MPC horizon

## Simulation setup ####################################################################

# Check that the initial and final positions are admissible for collision avoidance
init_ok = geometry.check_positions(
    x_init,
    d_min=mpc.d_min,
    dynamics=mpc.dynamics,
    u_max=mpc.u_max,
    dt=mpc.dt,
    verbose=True,
)
goal_ok = geometry.check_positions(
    x_goal,
    d_min=mpc.d_min,
    dynamics=mpc.dynamics,
    u_max=mpc.u_max,
    dt=mpc.dt,
    verbose=True,
)
if not init_ok:
    raise ValueError("Initial positions are not admissible for collision avoidance.")
if not goal_ok:
    raise ValueError("Goal positions are not admissible for collision avoidance.")

# Setup robotarium
if USE_ROBOTARIUM is True:

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

    # Initialize robotarium instance
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
time: np.ndarray = np.arange(0, T + DT, DT)

# Initialize cost weights
alpha_g: float = ALPHA_G
alpha_w: np.ndarray = ALPHA_W * (np.ones([m, m]) - np.eye(m))  # default constant

# Initialize winding number helper variables
theta: np.ndarray = np.zeros((m, m))  # relative headings between the agents
theta_prev: np.ndarray = invariants.relative_headings(x_init)  # prev relative headings
w_curr: np.ndarray = np.zeros((m, m))  # current winding numbers between the agents

# Initialize progress variables
tau: float = 0  # progress variable for winding-based progress
tau_mat: np.ndarray = np.zeros(len(time))  # to save tau over time for plotting
tau_i: np.ndarray = np.zeros(m) if mpc.architecture == "distributed" else None
tau_i_mat: np.ndarray = (
    np.zeros((len(time), m)) if mpc.architecture == "distributed" else None
)  # to save tau_i over time for plotting

# Initialize matrices to store simulation data
trajectories: np.ndarray = np.zeros((len(time), mpc.n_x, m))  # realized trajectories
cost_mat: np.ndarray = np.zeros((len(time), 4, m))  # cost for each agent (4 terms)
w_curr_mat: np.ndarray = np.zeros((len(time), m, m))  # save current WN for plotting
w_target_resampled: np.ndarray = np.zeros((len(time), m, m))  # resampled for plotting
t_sol_mat: np.ndarray = np.zeros((len(time), m))  # solution time for each agent

# Run simulation loop
for step, t in enumerate(time):

    # 1a. Update time varying weights
    if USE_TIME_VARYING_WEIGHTS is True:
        if PROGRESS_STRATEGY == "uniform_time":
            alpha_g = (
                ALPHA_G * (t / T) ** EXP_GOAL
            )  # increase goal cost weight over time
        elif PROGRESS_STRATEGY == "winding_progress":
            alpha_g = (
                ALPHA_G * (tau / 1.0) ** EXP_GOAL
            )  # increase goal cost weight over braid progress
        alpha_w = ALPHA_W * invariants.compute_winding_weights(
            np.array([M[i].x for i in range(m)]),
            d_threshold=None,
        )

    # 1b. Compute target winding numbers
    if PROGRESS_STRATEGY == "uniform_time":
        tau_target = min([(t + K * DT) / T, 1])  # cap target winding at end of braid
        w_target = windings_target[int(tau_target * (n_windings - 1)), :, :]
    elif PROGRESS_STRATEGY == "winding_progress":
        # Compute tau (progress variable) from current winding numbers
        if mpc.architecture == "distributed":
            for i in range(m):
                tau_i[i], _ = invariants.estimate_tau(
                    agent_idx=i,
                    w_measured=w_curr,
                    w_reference=windings_target,
                    tau_prev=tau,
                    delta_tau_max=delta_tau_max,
                    weights=alpha_w,
                )
            if PROGRESS_STRATEGY_DISTRIBUTED == "min":
                tau = tau_i.min()
            elif PROGRESS_STRATEGY_DISTRIBUTED == "mean":
                tau = tau_i.mean()
            else:
                raise ValueError(
                    f"Invalid distributed progress strategy: "
                    f"{PROGRESS_STRATEGY_DISTRIBUTED}"
                )
            tau_i_mat[step, :] = tau_i  # save individual tau_i for plotting
        else:  # centralized case
            tau, _ = invariants.estimate_tau(
                agent_idx=None,
                w_measured=w_curr,
                w_reference=windings_target,
                tau_prev=tau,
                delta_tau_max=delta_tau_max,
                weights=alpha_w,
            )

        # Find tau_target from tau and compute corresponding w_target
        tau_mat[step] = tau  # save tau for plotting
        tau_target = min(tau + delta_tau_max, 1)  # cap target winding at end of braid
        w_target = windings_target[int(tau_target * (n_windings - 1)), :, :]
    else:
        raise ValueError(f"Invalid progress strategy: {PROGRESS_STRATEGY}")
    w_target_resampled[step, :, :] = w_target  # save target WN for plotting

    # 2. Solve MPC problem
    if mpc.architecture == "distributed":

        # Collect predicted trajectories of all agents
        for i in range(m):
            if M[i].x_opt is None:  # 1st time step
                x_pred[:, :, i] = M[i].x  # use constant state as predicted trajectory
            else:
                x_pred[:-1, :, i] = M[i].x_opt[1:]
                x_pred[-1, :, i] = M[i].x_opt[-1]  # repeat last to match dimensions

        # Solve control problem for each agent in parallel
        for i in range(m):

            # Update dynamic weigths
            mpc.set_alpha_g(alpha_g)  # same for all agents
            mpc.set_alpha_w(np.delete(alpha_w[i], i, axis=0))  # local for each agent

            # Solve local MPC problem
            (u_opt, x_opt, cost, t_sol) = mpc.solve(
                x_0=M[i].x,
                x_goal=M[i].x_goal,
                x_pred=np.delete(x_pred, i, axis=2),
                x_prev=x_pred[:, :, i],
                w_curr=np.delete(w_curr[i], i, axis=0),
                w_target=np.delete(w_target[i], i, axis=0),
                sol_prev=M[i].sol,
                use_warm_start=True,
            )
            M[i].sol = mpc.sol  # save solution in agent object
            M[i].u_opt = u_opt
            M[i].x_opt = x_opt
            M[i].cost = cost
            M[i].t_sol = t_sol

            # Store individual components of the cost function and solve time
            if DEBUG is True:
                _, cost_g, cost_u, cost_w = mpc.check_cost()
                M[i].cost_u = cost_u
                M[i].cost_g = cost_g
                M[i].cost_w = cost_w
                cost_mat[step, 0, i] = cost
                cost_mat[step, 1, i] = cost_g
                cost_mat[step, 2, i] = cost_u
                cost_mat[step, 3, i] = cost_w
                t_sol_mat[step, i] = t_sol

            # Plot hyperplanes (requires manual selection of timestep and agent index)
            if (
                DEBUG_HYPERPLANES is True
                and mpc.architecture == "distributed"
                and step == 100
                and i == 0
            ):
                plot_debug.plot_hyperplanes(
                    mpc,
                    figsize=np.array([20, 20]),
                    show_solution=True,
                    show_legend=True,
                    show=True,
                    block=False,
                )

    # Centralized MPC controller
    elif mpc.architecture == "centralized":
        # Collect initial states of all agents
        x_0 = [M[i].x for i in range(m)]
        x_0 = np.array(x_0).T  # reshape to m x n_x

        # Update weights
        mpc.set_alpha_g(alpha_g)
        mpc.set_alpha_w(alpha_w)

        # Solve centralized MPC problem
        (u_opt, x_opt, cost, t_sol) = mpc.solve(
            x_0=x_0,
            x_goal=x_goal,
            w_curr=w_curr,
            w_target=w_target,
            use_warm_start=True,
        )

        # Save solution in agent objects
        for i in range(m):
            M[i].u_opt = u_opt[i]
            M[i].x_opt = x_opt[i]
            M[i].cost = cost  # same cost for all agents in centralized MPC
            M[i].t_sol = t_sol  # same solution time for all agents in centralized MPC

        if DEBUG is True:
            _, cost_g, cost_u, cost_w = mpc.check_cost()
            for i in range(m):
                M[i].cost_u = cost_u  # same cost for all in centralized MPC
                M[i].cost_g = cost_g
                M[i].cost_w = cost_w
            cost_mat[step, 0, :] = cost
            cost_mat[step, 1, :] = cost_g
            cost_mat[step, 2, :] = cost_u
            cost_mat[step, 3, :] = cost_w
            t_sol_mat[step, :] = t_sol  # same t_sol for all agents

    # Sanity check: verify that the predicted trajectories do not lead to collisions
    if DEBUG is True:
        for k in range(K):
            for i in range(m):
                for j in range(i + 1, m):
                    dist_ij = np.linalg.norm(M[i].x_opt[k, :2] - M[j].x_opt[k, :2])
                    if dist_ij < mpc.d_min:
                        if (
                            abs(dist_ij - mpc.d_min) > 1e-3
                        ):  # allow small numerical errors
                            print(
                                f"Warning: predicted trajectories of agents {i} and  "
                                f"{j} are too close at time step {k} "
                                f"(distance: {dist_ij:.4e})"
                            )

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
        for _ in range(int(DT / r.time_step)):
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

    # 4. Update current winding numbers
    theta = invariants.relative_headings(np.array([M[i].x for i in range(m)]).T)
    w_curr = w_curr + 1 / (2 * np.pi) * invariants.angle_diff(theta, theta_prev)
    w_curr_mat[step, :, :] = w_curr  # save w_curr for plotting
    theta_prev = theta

    # 5. Save data for plotting
    if DEBUG is True:
        prediction_error = np.linalg.norm(M[i].x_opt[1, :2] - M[i].x[:2])
        if prediction_error > 1e-3:  # allow small numerical errors
            print(
                f"Warning: large prediction error for agent {i} at time {t:.2f}s "
                f"(error: {prediction_error:.4e})"
            )
    for i in range(m):
        trajectories[step, :, i] = M[i].x

    # 6. Print debug info
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
                f"cost: {M[i].cost:.2e}",
                f"({M[i].cost_g:.2e}, {M[i].cost_u:.2e}, {M[i].cost_w:.2e})",
            )

## Compute stats #######################################################################
if DEBUG is True:
    t_sol_avg = np.mean(t_sol_mat, axis=0)
    t_sol_std = np.std(t_sol_mat, axis=0)
    t_sol_max = np.max(t_sol_mat, axis=0)

    if mpc.architecture == "distributed":
        print("\nAverage solution time for each agent:")
        for i in range(m):
            print(
                f"Agent {i}: {t_sol_avg[i]:.2f}s (std: {t_sol_std[i]:.2f}s, "
                f"max: {t_sol_max[i]:.2f}s)"
            )
    else:
        print(
            f"\nAverage solution time: {t_sol_avg[0]:.2f}s "
            f"(std: {t_sol_std[0]:.2f}s, max: {t_sol_max[0]:.2f}s)"
        )

## Evaluate results and show plots #####################################################

# Plot realized trajectories in space-time domain
plot.plot_paths_3d(
    trajectories[:, :2, :],
    x_lims=np.array(data["x_lims"]),
    y_lims=np.array(data["y_lims"]),
    normalize=False,
    show=False,
)

# Plot cost over time
plot.plot_cost(cost_mat, time, show=False)

# Plot realized winding numbers vs the target ones
plot.plot_windings(windings_target, range(windings_target.shape[0]), show=False)
plot.plot_windings(w_curr_mat, time, windings_ref=w_target_resampled, show=False)
if PROGRESS_STRATEGY == "winding_progress" and mpc.architecture == "distributed":
    plot.plot_tau(tau_mat, time, tau_i=tau_i_mat, show=False)
elif PROGRESS_STRATEGY == "winding_progress" and mpc.architecture == "centralized":
    plot.plot_tau(tau_mat, time, show=False)

# Show all plots (blocking)
if SHOW_PLOTS is True:

    def _close_all(event):
        if event.key in ("q", "escape"):
            plt.close("all")

    for num in plt.get_fignums():
        fig = plt.figure(num)
        fig.canvas.mpl_connect("key_press_event", _close_all)

    print("\nScript terminated. Press 'q' or 'escape' to close all plots.")

    plt.ioff()
    plt.show(block=True)

## Terminate script ####################################################################

# Call robotarium termination function
if USE_ROBOTARIUM is True:
    r.call_at_scripts_end()
