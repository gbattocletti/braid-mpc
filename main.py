import matplotlib.pyplot as plt
import numpy as np
import rps.robotarium as rb

# pylint: disable=unused-import
from core import agent, mpc_centralized, mpc_distributed
from data import grids_1, grids_2
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

## Controllers and agents setup ########################################################
# Initialize controller's properties
dt = 0.5  # s
K = 20  # time steps

# Create agents
M = [agent.Agent(i) for i in range(m)]
for i in range(m):
    M[i].dt = dt
    M[i].x = x_init[i, :]
    M[i].x_goal = x_goal[i, :]

# Initialize MPC controller
# NOTE: when using the robotarium, the velocity limits (u limits) must currently be
# scaled manually to match the robotarium's velocity limits, depending on the scaling
# factor used between the robotarium environment size and the 'real world' environment
# size. Otherwise, the robot's velocity will be different than expected.
mpc = mpc_distributed.DistributedMPC()
# mpc = mpc_centralized.CentralizedMPC()
mpc.dt = dt
mpc.K = K
mpc.m = m
mpc.alpha_u = 0
mpc.alpha_g = 1000.0
mpc.alpha_w = 0
mpc.R = np.diag([1, 1])
mpc.u_min = np.array([-0.625, -0.625])
mpc.u_max = np.array([0.625, 0.625])
mpc.x_min = np.array([0, 0])
mpc.x_max = np.array([10, 10])
mpc.d_min = 0.05  # slightly larger than needed
mpc.initialize_ocp()

# Initialize helper variables
x_pred: np.ndarray | None
if mpc.architecture == "distributed":
    x_pred = np.zeros([K, 2, m])
elif mpc.architecture == "centralized":
    x_pred = None  # not needed for centralized MPC
    x_goal = np.array([M[i].x_goal for i in range(m)]).T  # reshape to 2 x m

# Initialize robotarium variables
v_vec = np.zeros((2, m))  # only used for robotarium

## Simulation setup ####################################################################
if USE_ROBOTARIUM is True:
    x_init_robotarium: np.ndarray = np.vstack(
        [
            robotarium_bridge.real2robotarium(
                x_init,
                [mpc.x_min[0][0], mpc.x_max[0][0]],
                [mpc.x_min[0][1], mpc.x_max[0][1]],
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

    # 1. Compute current and target winding numbers
    # NOTE: currently the time progresses constantly along the braid
    tau = t / T  # time n
    w_curr = windings[int(tau * (n_windings - 1)), :, :]
    tau_target = min((t + K * dt) / T, 1)  # cap target time at the end of the braid
    w_target = windings[int(tau_target * (n_windings - 1)), :, :]

    # 2. Solve MPC problem
    if mpc.architecture == "distributed":
        # Collect predicted trajectories of all agents
        # TODO: check dimensions and indexing
        for i in range(m):
            x_pred[0:, :, i] = M[i].x
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
        x_0 = np.array(x_0).T  # reshape to m x 2

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

    else:
        raise ValueError(f"Invalid architecture: {mpc.architecture}")

    # 3. Take one step in the environment and get new state
    if USE_ROBOTARIUM is True:
        for i in range(m):
            v_vec[:, i] = robotarium_bridge.real2robotarium(
                M[i].u_opt[0],
                [mpc.u_min[0][0], mpc.u_max[0][0]],
                [mpc.u_min[0][1], mpc.u_max[0][1]],
                coords_type="velocity",
            )

        # Simulate with Robotarium
        r.set_velocities(range(m), v_vec)
        r.step()

        # Get new states from Robotarium
        # TODO: check dimensions and indexing
        x_meas = r.get_poses()
        for i in range(m):
            M[i].x = robotarium_bridge.robotarium2real(
                x_meas[:, i],
                [mpc.x_min[0][0], mpc.x_max[0][0]],
                [mpc.x_min[0][1], mpc.x_max[0][1]],
                coords_type="position",
            )

    else:
        # Simulate step using agent's dynamics
        for i in range(m):
            M[i].step(M[i].u_opt[0])

    # 4. Print debug info
    # TODO: check dimensions and indexing
    if DEBUG is True:
        print(f"t: {t:5.2f}s", end="")
        for i in range(m):
            print(
                f"\ti: {i}, "
                f"x(k): [{M[i].x_opt[0]:5.2f},{M[i].x_opt[1]:5.2f}], "
                f"x(k+1): [{M[i].x[0]:5.2f},{M[i].x[1]:5.2f}], "
                f"u*(0|k): [{M[i].u_opt[0]: 4.2f},{M[i].u_opt[1]: 4.2f}], "
                f"t_sol: {M[i].t_sol:.2f}s, "
                f"cost: {M[i].cost:.2f}"
            )

# Call robotarium termination function
r.call_at_scripts_end()

## Evaluate results and show plots #####################################################
# TODO

# Show all plots
plt.show()
