import os

import numpy as np

from braid_controller.core import agent, mpc_centralized, mpc_distributed
from braid_controller.utils import geometry, invariants, io, weights
from braid_controller.utils.grid_controller import grid_controller
from braid_controller.visualization import plot

## Settings ############################################################################

# User-defined settings
experiments = [
    "grids_m5_1.yaml",
    "grids_m5_2.yaml",
    "grids_m10_1.yaml",
    "grids_m10_2.yaml",
    "grids_m10_3.yaml",
    "grids_m10_4.yaml",
    "grids_m10_5.yaml",
    "grids_m10_6.yaml",
]

controllers = [
    "distributed",
    "centralized",
    "grid",
]

# Simulation and controller's properties
dt: float = 0.25  # time step
t_end: float = 1000  # total simulation time (s) NOTE: much larger than actually needed
K: int = 21  # MPC horizon

# Cost function weights
d_min: float = 0.7  # minimum distance between agents
alpha_u: float = 0.05  # control weight
alpha_g: float = 1  # scaling factor for goal tracking cost
alpha_w: float = 20  # scaling factor for winding cost
coeff_s: float = 20  # sharpness of sigmoid function for time-varying weights
coeff_c: float = 0.9  # center of sigmoid function for time-varying weights [0,1]
u_max: np.ndarray = np.array([0.5, 0.5])  # maximum control input (m/s)
u_rate_max: np.ndarray = np.array([0.15, 0.15]) * dt  # max velocity change (a * dt)

## Preprocessing #######################################################################
np.random.seed(1312)  # Set random seed for reproducibility
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Move to script directory

for experiment in experiments:

    # Print experiment info
    print("####################################################################")
    print(f"Running experiment: {experiment}")

    # Load input data
    data: dict = io.load_yaml("data/" + experiment)

    # Create output directory
    output_dir: str = "results/" + experiment.split(".", maxsplit=1)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Create set of agents
    m: int = data["m"]
    M = [agent.Agent(i) for i in range(m)]

    # Check initial and final positions (dummy heading added to get correct dimension)
    x_init: np.ndarray = np.array(data["x_init"]).T
    x_goal: np.ndarray = np.array(data["x_goal"]).T
    x_init = np.vstack([x_init, np.zeros([1, m])]) if x_init.shape[0] == 2 else x_init
    x_goal = np.vstack([x_goal, np.zeros([1, m])]) if x_goal.shape[0] == 2 else x_goal
    if not geometry.check_positions(
        x_init,
        d_min=d_min,
        dynamics="single_integrator",
        u_max=np.array([u_max]),
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
        u_max=np.array([u_max]),
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
    w_target_full: np.ndarray | None = None
    n_windings: int | None = None
    if "braid" in data:
        pass  # TODO compute winding_targets directly from braid
        # braid = np.array(data["braid"])
        # n_generators_braid = braid.shape[0]  # number of generators in the braid word
        # n_generators = n_generators_braid
        # w_target_full = invariants.braid2windings(braid, m)  --> to implement
    if "grids" in data:
        grids = np.array(data["grids"])
        n_generators_grid = grids.shape[0]  # number of generators in the specification
        if w_target_full is None:
            paths = invariants.grids2paths(grids)
            plot_grid_spec_3d, _ = plot.plot_paths_3d(
                paths,
                x_lims=np.array(
                    [
                        data["x_lims"][0] - 1,
                        data["x_lims"][1] + 1,
                    ]
                ),
                y_lims=np.array(
                    [
                        data["y_lims"][0] - 1,
                        data["y_lims"][1] + 1,
                    ]
                ),
                normalize=False,
            )
            w_target_full = invariants.paths2windings(
                paths,
                upscale_factor=100,
                intermediate_shape="linear",
            )
            n_windings = w_target_full.shape[0]
            n_generators = n_generators_grid

    # Iterate over controllers and run simulations
    for controller in controllers:

        # Print start of control loop
        print(f"\nRunning {controller} controller.")

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

        # Reset MPC parameters, winding numbers, and progress variables
        horizon: np.ndarray = np.arange(0, K + 1)  # [0, ..., K], K+1 elements
        alpha_g_k: float = alpha_g
        alpha_w_k: np.ndarray = alpha_w * (np.ones([m, m]) - np.eye(m))
        theta: np.ndarray = np.zeros((m, m))
        theta_prev: np.ndarray = invariants.relative_headings(x_init)
        w_curr: np.ndarray = np.zeros((m, m))
        tau: float = 0
        delta_tau: float = (
            2 / (n_generators * np.pi) * np.arcsin(np.linalg.norm(u_max) * dt / d_min)
        )
        delta_tau_K: float = K * delta_tau  # max change in tau over one MPC horizon

        # Initialize matrices to save data for plotting
        goal_reached: bool = False
        step: int = 0
        time: np.ndarray = np.arange(0, t_end + dt, dt)
        n_time: int = len(time)
        trajectories: np.ndarray = np.zeros((len(time), 2, m))
        actuation: np.ndarray = np.zeros((len(time), 2, m))
        cost_mat: np.ndarray = np.zeros((len(time), 4, m))
        tau_mat: np.ndarray = np.zeros(len(time))
        w_curr_mat: np.ndarray = np.zeros((len(time), m, m))
        w_target_mat: np.ndarray = np.zeros((len(time), m, m))
        t_sol_mat: np.ndarray = np.zeros((len(time), m))

        # Simulate controllers
        if controller == "centralized":

            # Initialize MPC controller
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

            # Disable helper variables in centralized mpc
            x_pred = None
            x_prev = None

            # Run control loop
            while goal_reached is False and step < n_time:

                # 1. Update time-varying weights based on tau
                alpha_g_k = alpha_g * weights.sigmoid(
                    tau,
                    coeff_sharpness=coeff_s,
                    coeff_center=coeff_s,
                )
                alpha_w_k = (
                    alpha_w  # constant weight coefficient
                    * (
                        1
                        - weights.sigmoid(
                            tau,
                            coeff_sharpness=coeff_s,
                            coeff_center=coeff_c,
                        )
                    )  # sigmoid-based weight (decreasing with tau to prioritize goal)
                    * invariants.compute_winding_weights(
                        np.array([M[i].x for i in range(m)]),
                        d_threshold=None,
                    )  # distance-based weights
                )

                # 2. Estimate tau
                tau, _ = invariants.estimate_tau(
                    agent_idx=None,
                    w_measured=w_curr,
                    w_reference=w_target_full,
                    tau_prev=tau,
                    delta_tau_max=delta_tau,
                    weights=alpha_w_k,
                )
                tau_mat[step] = tau
                tau_target = np.clip(tau + horizon * delta_tau, 0, 1)  # sequence of tau
                w_target = w_target_full[
                    (tau_target * (n_windings - 1)).astype(int), :, :
                ]
                w_target_mat[step, :, :] = w_target[-1, :, :]

                # 3. Solve centralized MPC problem
                x_0 = [M[i].x for i in range(m)]
                x_0 = np.array(x_0).T  # reshape to m x n_x
                mpc.set_alpha_g(alpha_g_k)
                mpc.set_alpha_w(alpha_w_k)
                u_opt, x_opt, cost, t_sol = mpc.solve(
                    x_0=x_0,
                    x_goal=x_goal,
                    w_curr=w_curr,
                    w_target=w_target,
                    use_warm_start=True,
                )
                for i in range(m):
                    M[i].u_opt = u_opt[i]
                    M[i].x_opt = x_opt[i]
                    M[i].cost = cost
                    M[i].t_sol = t_sol
                    t_sol_mat[step, i] = t_sol
                _, cost_g, cost_u, cost_w = mpc.check_cost()
                cost_mat[step, 0, :] = cost
                cost_mat[step, 1, :] = cost_g
                cost_mat[step, 2, :] = cost_u
                cost_mat[step, 3, :] = cost_w

                # 4. Execute control action
                for i in range(m):
                    M[i].step(M[i].u_opt[0])
                    trajectories[step, :, i] = M[i].x[:2]
                    actuation[step, :, i] = M[i].u_opt[0] * dt

                # 5. Update current winding numbers
                theta = invariants.relative_headings(
                    np.array([M[i].x for i in range(m)]).T
                )
                w_curr = w_curr + 1 / (2 * np.pi) * invariants.angle_diff(
                    theta, theta_prev
                )
                w_curr_mat[step, :, :] = w_curr
                theta_prev = theta

                # 6. Check if sim is completed or move to next time step
                if tau > 0.9 and np.all(
                    np.linalg.norm(
                        np.array([M[i].x[:2] for i in range(m)]) - x_goal[:2, :].T,
                        axis=-1,
                    )
                    < 0.1
                ):
                    goal_reached = True
                else:
                    step += 1

            # Cut matrices to correct length
            time = time[:step]
            trajectories = trajectories[:step, :, :]
            cost_mat = cost_mat[:step, :, :]
            tau_mat = tau_mat[:step]
            w_curr_mat = w_curr_mat[:step, :, :]
            w_target_mat = w_target_mat[:step, :, :]
            t_sol_mat = t_sol_mat[:step, :]

            # Compute stats
            t_sol_avg = np.mean(t_sol_mat, axis=0)
            t_sol_std = np.std(t_sol_mat, axis=0)
            t_sol_max = np.max(t_sol_mat, axis=0)
            traj_len = np.zeros(m)
            for i in range(m):
                traj_len[i] = np.sum(
                    np.linalg.norm(np.diff(trajectories[:, :, i], axis=0), axis=-1)
                )
            traj_mean = np.mean(traj_len)
            traj_std = np.std(traj_len)
            traj_min = np.min(traj_len)
            traj_max = np.max(traj_len)
            force = np.linalg.norm(actuation * dt, axis=1)
            force_max = np.max(force)
            force_mean = np.mean(force)
            force_std = np.std(force)
            force_tot = np.sum(force)

            # Print & save stats
            print("Centralized MPC")
            print(f"\tTotal time: {time[-1]}")
            print("Solver time:")
            print(
                f"\t{t_sol_avg[0]:.4f}s "
                f"(std: {t_sol_std[0]:.4f}s, max: {t_sol_max[0]:.4f}s)"
            )
            print("Paths:")
            print(
                f"\t{traj_mean:.4f} +- {traj_std:.4f} "
                f"({traj_min:.4f} -- {traj_max:.4f})"
            )
            print("Forces:")
            print(
                f"\t{force_mean:.4f} +- {force_std:.4f}"
                f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
            )
            with open(
                os.path.join(output_dir, "c_stats.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("Centralized MPC\n")
                f.write(f"Total time: {time[-1]}\n")
                f.write(
                    f"Solver time (global): {t_sol_avg[0]:.4f}s "
                    f"(std: {t_sol_std[0]:.4f}s, max: {t_sol_max[0]:.4f}s)"
                )
                f.write(
                    f"Paths: {traj_mean:.4f} +- {traj_std:.4f} "
                    f"({traj_min:.4f} -- {traj_max:.4f})"
                )
                f.write(
                    f"Forces: {force_mean:.4f} +- {force_std:.4f}"
                    f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
                )
            np.savez(
                os.path.join(output_dir, "c_data.txt"),
                dt=dt,
                K=K,
                m=m,
                d_min=d_min,
                alpha_u=alpha_u,
                alpha_g=alpha_g,
                alpha_w=alpha_w,
                coeff_s=coeff_s,
                coeff_c=coeff_c,
                u_max=u_max,
                u_rate_max=u_rate_max,
                x_init=x_init,
                x_goal=x_goal,
                x_lim=data["x_lims"],
                y_lim=data["y_lims"],
                w_target=w_target_full,
                time=time,
                trajectories=trajectories,
                actuation=actuation,
                cost_mat=cost_mat,
                tau_mat=tau_mat,
                w_curr_mat=w_curr_mat,
                w_target_mat=w_target_mat,
                t_sol_mat=t_sol_mat,
            )

            # Generate and save plots
            fig_paths, _ = plot.plot_paths_3d(
                trajectories[:, :2, :],
                time=time,
                x_lims=np.array(data["x_lims"]),
                y_lims=np.array(data["y_lims"]),
                normalize=False,
            )
            fig_cost, _ = plot.plot_cost(cost_mat, time)
            fig_w_target, _ = plot.plot_windings(
                w_target_full, range(w_target_full.shape[0])
            )
            fig_w_curr, _ = plot.plot_windings(w_curr_mat, time)
            fig_tau, _ = plot.plot_tau(tau_mat, time)
            fig_paths.savefig(os.path.join(output_dir, "c_paths.png"), dpi=900)
            fig_cost.savefig(os.path.join(output_dir, "c_cost.png"), dpi=900)
            fig_w_target.savefig(os.path.join(output_dir, "c_w_target.png"), dpi=900)
            fig_w_curr.savefig(os.path.join(output_dir, "c_w_curr.png"), dpi=900)
            fig_tau.savefig(os.path.join(output_dir, "c_tau.png"), dpi=900)

        elif controller == "distributed":
            # Initialize MPC controller
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

            # Initialize helper variables to store trajectory predictions
            x_pred = np.zeros([K + 1, mpc.n_x, m])  # shape: (K+1, n_x, m)
            x_prev = np.zeros([K + 1, mpc.n_x])  # shape: (K+1, n_x).
            tau_i: np.ndarray = np.zeros(m)
            tau_i_mat: np.ndarray = np.zeros((len(time), m))

            # Run control loop
            while goal_reached is False and step < n_time:

                # 1. Update time-varying weights based on tau
                alpha_g_k = alpha_g * weights.sigmoid(
                    tau,
                    coeff_sharpness=coeff_s,
                    coeff_center=coeff_c,
                )
                alpha_w_k = (
                    alpha_w  # constant weight coefficient
                    * (
                        1
                        - weights.sigmoid(
                            tau,
                            coeff_sharpness=coeff_s,
                            coeff_center=coeff_c,
                        )
                    )  # sigmoid-based weight (decreasing with tau to prioritize goal)
                    * invariants.compute_winding_weights(
                        np.array([M[i].x for i in range(m)]),
                        d_threshold=None,
                    )  # distance-based weights
                )

                # 2. Estimate tau + consensus
                for i in range(m):
                    tau_i[i], _ = invariants.estimate_tau(
                        agent_idx=i,
                        w_measured=w_curr,
                        w_reference=w_target_full,
                        tau_prev=tau,
                        delta_tau_max=delta_tau,
                        weights=alpha_w_k,
                    )
                    tau = np.mean(tau_i)
                tau_i_mat[step, :] = tau_i
                tau_mat[step] = tau
                tau_target = np.clip(tau + horizon * delta_tau, 0, 1)  # sequence of tau
                w_target = w_target_full[
                    (tau_target * (n_windings - 1)).astype(int), :, :
                ]
                w_target_mat[step, :, :] = w_target[-1, :, :]

                # 3. Update predicted trajectories for each agent
                for i in range(m):
                    if M[i].x_opt is None:  # 1st time step
                        x_pred[:, :, i] = M[i].x
                    else:
                        x_pred[:-1, :, i] = M[i].x_opt[1:]
                        x_pred[-1, :, i] = M[i].x_opt[-1]

                # 4. Solve distributed MPC for each agent
                for i in range(m):
                    mpc.set_alpha_g(alpha_g_k)
                    mpc.set_alpha_w(np.delete(alpha_w_k[i], i, axis=0))
                    u_opt, x_opt, cost, t_sol = mpc.solve(
                        x_0=M[i].x,
                        x_goal=M[i].x_goal,
                        x_pred=np.delete(x_pred, i, axis=2),
                        x_prev=x_pred[:, :, i],
                        w_curr=np.delete(w_curr[i], i, axis=0),
                        w_target=np.delete(w_target[:, i, :], i, axis=1),
                        sol_prev=M[i].sol,
                        use_warm_start=True,
                        u_prev=(
                            np.reshape(M[i].v[:2], (1, mpc.n_u))
                            if mpc.use_u_rate_constraints is True
                            else None
                        ),
                    )
                    M[i].sol = mpc.sol  # save solution in agent object
                    M[i].u_opt = u_opt
                    M[i].x_opt = x_opt
                    M[i].cost = cost
                    M[i].t_sol = t_sol
                    t_sol_mat[step, :] = t_sol
                    _, cost_g, cost_u, cost_w = mpc.check_cost()
                    cost_mat[step, 0, i] = cost
                    cost_mat[step, 1, i] = cost_g
                    cost_mat[step, 2, i] = cost_u
                    cost_mat[step, 3, i] = cost_w

                # 5. Execute control action
                for i in range(m):
                    M[i].step(M[i].u_opt[0])
                    trajectories[step, :, i] = M[i].x[:2]
                    actuation[step, :, i] = M[i].u_opt[0] * dt

                # 6. Update current winding numbers
                theta = invariants.relative_headings(
                    np.array([M[i].x for i in range(m)]).T
                )
                w_curr = w_curr + 1 / (2 * np.pi) * invariants.angle_diff(
                    theta, theta_prev
                )
                w_curr_mat[step, :, :] = w_curr
                theta_prev = theta

                # 6. Check if sim is completed or move to next time step
                if tau > 0.9 and np.all(
                    np.linalg.norm(
                        np.array([M[i].x[:2] for i in range(m)]) - x_goal[:2, :].T,
                        axis=-1,
                    )
                    < 0.2  # CHECKME may need tuning
                ):
                    goal_reached = True
                else:
                    step += 1

            # Cut matrices to correct length
            time = time[:step]
            trajectories = trajectories[:step, :, :]
            cost_mat = cost_mat[:step, :, :]
            tau_mat = tau_mat[:step]
            tau_i_mat = tau_i_mat[:step, :]
            w_curr_mat = w_curr_mat[:step, :, :]
            w_target_mat = w_target_mat[:step, :, :]
            t_sol_mat = t_sol_mat[:step, :]

            # Compute stats
            t_sol_avg = np.mean(t_sol_mat, axis=0)
            t_sol_std = np.std(t_sol_mat, axis=0)
            t_sol_max = np.max(t_sol_mat, axis=0)
            traj_len = np.zeros(m)
            for i in range(m):
                traj_len[i] = np.sum(
                    np.linalg.norm(np.diff(trajectories[:, :, i], axis=0), axis=-1)
                )
            traj_mean = np.mean(traj_len)
            traj_std = np.std(traj_len)
            traj_min = np.min(traj_len)
            traj_max = np.max(traj_len)
            force = np.linalg.norm(actuation * dt, axis=1)
            force_max = np.max(force)
            force_mean = np.mean(force)
            force_std = np.std(force)
            force_tot = np.sum(force)

            # Print & save stats
            print("Distributed MPC")
            print(f"\tTotal time: {time[-1]}")
            print("Solver time:")
            for i in range(m):
                print(
                    f"\tAgent {i}: {t_sol_avg[i]:.4f}s (std: {t_sol_std[i]:.4f}s, "
                    f"max: {t_sol_max[i]:.4f}s)"
                )
            print("Solver time (global):")
            print(
                f"\t{np.mean(t_sol_mat):.4f}s (std: {np.std(t_sol_mat):.4f}s, "
                f"max: {np.max(t_sol_mat):.4f}s)"
            )
            print("Paths:")
            print(
                f"\t{traj_mean:.4f} +- {traj_std:.4f} "
                f"({traj_min:.4f} -- {traj_max:.4f})"
            )
            print("Forces:")
            print(
                f"{force_mean:.4f} +- {force_std:.4f}"
                f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
            )
            with open(
                os.path.join(output_dir, "d_stats.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("Distributed MPC\n")
                f.write(f"Total time: {time[-1]}\n")
                for i in range(m):
                    f.write(
                        f"Solver time agent {i}: {t_sol_avg[i]:.4f}s "
                        f"(std: {t_sol_std[i]:.4f}s, max: {t_sol_max[i]:.4f}s)"
                    )
                f.write(
                    f"Solver time (global): {np.mean(t_sol_mat):.4f} "
                    f"(std: {np.std(t_sol_mat):.4f}, max: {np.max(t_sol_mat):.4f})"
                )
                f.write(
                    f"Paths: {traj_mean:.4f} +- {traj_std:.4f} "
                    f"({traj_min:.4f} -- {traj_max:.4f})"
                )
                f.write(
                    f"Forces: {force_mean:.4f} +- {force_std:.4f}"
                    f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
                )
            np.savez(
                os.path.join(output_dir, "d_data.txt"),
                dt=dt,
                K=K,
                m=m,
                d_min=d_min,
                alpha_u=alpha_u,
                alpha_g=alpha_g,
                alpha_w=alpha_w,
                coeff_s=coeff_s,
                coeff_c=coeff_c,
                u_max=u_max,
                u_rate_max=u_rate_max,
                x_init=x_init,
                x_goal=x_goal,
                x_lim=data["x_lims"],
                y_lim=data["y_lims"],
                w_target=w_target_full,
                time=time,
                trajectories=trajectories,
                actuation=actuation,
                cost_mat=cost_mat,
                tau_mat=tau_mat,
                tau_i_mat=tau_i_mat,
                w_curr_mat=w_curr_mat,
                w_target_mat=w_target_mat,
                t_sol_mat=t_sol_mat,
            )

            # Generate and save plots
            fig_paths, _ = plot.plot_paths_3d(
                trajectories[:, :2, :],
                time=time,
                x_lims=np.array(data["x_lims"]),
                y_lims=np.array(data["y_lims"]),
                normalize=False,
            )
            fig_cost, _ = plot.plot_cost(cost_mat, time)
            fig_w_target, _ = plot.plot_windings(
                w_target_full, range(w_target_full.shape[0])
            )
            fig_w_curr, _ = plot.plot_windings(w_curr_mat, time)
            fig_tau, _ = plot.plot_tau(tau_mat, time, tau_i=tau_i_mat)
            fig_paths.savefig(os.path.join(output_dir, "d_paths.png"), dpi=900)
            fig_cost.savefig(os.path.join(output_dir, "d_cost.png"), dpi=900)
            fig_w_target.savefig(os.path.join(output_dir, "d_w_target.png"), dpi=900)
            fig_w_curr.savefig(os.path.join(output_dir, "d_w_curr.png"), dpi=900)
            fig_tau.savefig(os.path.join(output_dir, "d_tau.png"), dpi=900)

        elif controller == "grid":

            # Initialize grid controller
            side_x = data["x_lims"][1] - data["x_lims"][0]
            side_y = data["y_lims"][1] - data["y_lims"][0]
            d_x = side_x / m
            d_y = side_y / m
            if d_x < d_min or d_y < d_min:
                raise ValueError(
                    "Grid spacing is too small for collision avoidance. "
                    "Increase the area or reduce the number of agents."
                )
            grid_points_x = np.arange(d_x / 2, side_x, d_x)
            grid_points_y = np.arange(d_y / 2, side_y, d_y)

            # Run control loop
            grid_gen_idx: int = 0
            targets: np.ndarray = np.zeros((m, 2))
            move_to_goal: bool = False
            while goal_reached is False and step < n_time:

                # Select target
                if move_to_goal is False:
                    for i in range(m):
                        row, col = np.where(grids[grid_gen_idx] == i + 1)
                        targets[i, :] = np.array(
                            [
                                grid_points_x[col][0],
                                grid_points_y[row][0],
                            ]  # CHECKME is row and col correct?
                        )
                else:
                    for i in range(m):
                        targets[i, :] = x_goal[:2, i]  # use goal as next target
                tau_mat[step] = np.max([0, grid_gen_idx - 1]) / n_generators_grid

                # Plan motion for each robot
                for i in range(m):
                    x_opt, u_opt, cost, t_sol = grid_controller(
                        x_init=M[i].x[:2],
                        x_goal=targets[i, :],
                        K=K,
                        dt=dt,
                        u_max=u_max,
                        u_rate_max=u_rate_max,
                        u_prev=M[i].v[:2],  # note: v = u (without multiplication by dt)
                    )
                    M[i].u_opt = u_opt
                    M[i].x_opt = x_opt
                    M[i].cost = cost
                    M[i].t_sol = t_sol
                    t_sol_mat[step, i] = t_sol
                    cost_mat[step, :, i] = cost

                # Execute control actions
                for i in range(m):
                    M[i].step(M[i].u_opt[:, 0])
                    trajectories[step, :, i] = M[i].x[:2]
                    actuation[step, :, i] = M[i].u_opt[:, 0] * dt

                # Advance progress if agent is close enough to target
                x_all = np.array([M[i].x[:2] for i in range(m)])
                distances = np.linalg.norm(x_all - targets, axis=-1)
                if np.all(distances < 0.1):
                    if grid_gen_idx < n_generators_grid - 1:
                        grid_gen_idx += 1  # move to next index
                    else:
                        move_to_goal = True

                # 6. Check if sim is completed or move to next time step
                if move_to_goal is True and np.all(
                    np.linalg.norm(
                        np.array([M[i].x[:2] for i in range(m)]) - x_goal[:2, :].T,
                        axis=-1,
                    )
                    < 0.1
                ):
                    goal_reached = True
                else:
                    step += 1

            # Cut matrices to correct length
            time = time[:step]
            trajectories = trajectories[:step, :, :]
            cost_mat = cost_mat[:step, :, :]
            tau_mat = tau_mat[:step]
            w_curr_mat = w_curr_mat[:step, :, :]
            w_target_mat = w_target_mat[:step, :, :]
            t_sol_mat = t_sol_mat[:step, :]

            # Compute stats
            t_sol_avg = np.mean(t_sol_mat, axis=0)
            t_sol_std = np.std(t_sol_mat, axis=0)
            t_sol_max = np.max(t_sol_mat, axis=0)
            traj_len = np.zeros(m)
            for i in range(m):
                traj_len[i] = np.sum(
                    np.linalg.norm(np.diff(trajectories[:, :, i], axis=0), axis=-1)
                )
            traj_mean = np.mean(traj_len)
            traj_std = np.std(traj_len)
            traj_min = np.min(traj_len)
            traj_max = np.max(traj_len)
            force = np.linalg.norm(actuation * dt, axis=1)
            force_max = np.max(force)
            force_mean = np.mean(force)
            force_std = np.std(force)
            force_tot = np.sum(force)

            # Print & save stats
            print("Grid Controller")
            print(f"\tTotal time: {time[-1]}")
            print("Solver time:")
            for i in range(m):
                print(
                    f"\tAgent {i}: {t_sol_avg[i]:.4f}s (std: {t_sol_std[i]:.4f}s, "
                    f"max: {t_sol_max[i]:.4f}s)"
                )
            print("Solver time (global):")
            print(
                f"\t{np.mean(t_sol_mat):.4f}s (std: {np.std(t_sol_mat):.4f}s, "
                f"max: {np.max(t_sol_mat):.4f}s)"
            )
            print("Paths:")
            print(
                f"{traj_mean:.4f} +- {traj_std:.4f} "
                f"({traj_min:.4f} -- {traj_max:.4f})"
            )
            print("Forces:")
            print(
                f"{force_mean:.4f} +- {force_std:.4f}"
                f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
            )
            with open(
                os.path.join(output_dir, "g_stats.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("Grid Controller\n")
                f.write(f"Total time: {time[-1]}\n")
                print("Solver time:")
                for i in range(m):
                    f.write(
                        f"Solver time agent {i}: {t_sol_avg[i]:.4f}s "
                        f"(std: {t_sol_std[i]:.4f}s, max: {t_sol_max[i]:.4f}s)"
                    )
                f.write(
                    f"Solver time (global): {np.mean(t_sol_mat):.4f} "
                    f"(std: {np.std(t_sol_mat):.4f}, max: {np.max(t_sol_mat):.4f})"
                )
                f.write(
                    f"Paths: {traj_mean:.4f} +- {traj_std:.4f} "
                    f"({traj_min:.4f} -- {traj_max:.4f})"
                )
                f.write(
                    f"Force: {force_mean:.4f} +- {force_std:.4f}"
                    f"(max: {force_max:.4f}, tot: {force_tot:.4f})"
                )
            np.savez(
                os.path.join(output_dir, "d_data.txt"),
                dt=dt,
                K=K,
                m=m,
                d_min=d_min,
                alpha_u=alpha_u,
                alpha_g=alpha_g,
                alpha_w=alpha_w,
                coeff_s=coeff_s,
                coeff_c=coeff_c,
                u_max=u_max,
                u_rate_max=u_rate_max,
                x_init=x_init,
                x_goal=x_goal,
                x_lim=data["x_lims"],
                y_lim=data["y_lims"],
                w_target=w_target_full,
                time=time,
                trajectories=trajectories,
                actuation=actuation,
                cost_mat=cost_mat,
                tau_mat=tau_mat,
                w_curr_mat=w_curr_mat,
                w_target_mat=w_target_mat,
                t_sol_mat=t_sol_mat,
            )

            # Generate and save plots
            fig_paths, _ = plot.plot_paths_3d(
                trajectories[:, :2, :],
                time=time,
                x_lims=np.array(
                    [
                        data["x_lims"][0] - 1,
                        data["x_lims"][1] + 1,
                    ]
                ),
                y_lims=np.array(
                    [
                        data["y_lims"][0] - 1,
                        data["y_lims"][1] + 1,
                    ]
                ),
                normalize=False,
            )
            fig_paths.savefig(os.path.join(output_dir, "g_paths.png"), dpi=900)
