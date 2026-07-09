"""
Visualize the rosbag data logged by the px4-braid-mpc nodes.
"""

from pathlib import Path as FilePath

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.axes3d import Axes3D

from braid_controller.utils.braidlab_bridge import Braidlab
from braid_controller.visualization.plot import tab10_colors, tab60_colors

########################################################################################
# Load data
experiment: str = "grids_m5_1"  # NOTE: change to plot another file
folder = FilePath(__file__).resolve().parent / experiment

# Load npz data
npz_path: FilePath = folder / "d_data.npz"
if not npz_path.exists():
    raise FileNotFoundError(f"npz file not found: {npz_path}.")
with np.load(npz_path, allow_pickle=False) as npz:
    data: dict = {k: npz[k] for k in npz.files}
print(f"Loaded {npz_path.name}.")

########################################################################################
# Generate plot

# Set LaTeX font for plots
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams.update(
    {
        "axes.labelsize": 8,  # x / y / z axis labels
        "xtick.labelsize": 6,  # tick labels
        "ytick.labelsize": 6,
    }
)

# Generate figures
# 1. Target windings
figsize = np.array([5, 3])  # width, height in cm
fig_w_target: plt.Figure
ax_w_target: plt.Axes
fig_w_target, ax_w_target = plt.subplots(figsize=figsize / 2.54)
w_target: np.ndarray = data["w_target"]
n, m, _ = w_target.shape
n_lines = int(m * (m - 1) / 2)
if n_lines > 10:
    colors = tab60_colors(n_lines)
else:
    colors = tab10_colors(n_lines)
tau = np.linspace(0, 1, n)
line_idx: int = 0
for i in range(m):
    for j in range(m):
        if j >= i:
            continue  # only plot windings once (lower triangle of w_target)
        ax_w_target.plot(
            tau,
            w_target[:, i, j],
            linewidth=1.2,
            color=colors[line_idx],
            # label=f"w_{{{i}{j}}}",
            zorder=2,
        )
        line_idx += 1
ax_w_target.set_xlim(0, 1)
ax_w_target.set_ylim(w_target.min() - 0.5, w_target.max() + 0.5)
ax_w_target.grid(
    True,
    which="major",
    linestyle=":",
    color="gray",
    linewidth=0.5,
    zorder=1,
)
ax_w_target.grid(
    True,
    which="minor",
    linestyle=":",
    color="gray",
    linewidth=0.3,
    zorder=1,
)
ax_w_target.minorticks_on()
ax_w_target.set_xlabel(r"$\tau$")
ax_w_target.set_ylabel(r"$w_{i,j}$")
ax_w_target.grid(True)


# 2. Real windings
figsize = np.array([5, 3])  # width, height in cm
fig_w_real: plt.Figure
ax_w_real: plt.Axes
fig_w_real, ax_w_real = plt.subplots(figsize=figsize / 2.54)
w_real: np.ndarray = data["w_curr_mat"]
time: np.ndarray = data["time"]
n, m, _ = w_real.shape
n_lines = int(m * (m - 1) / 2)
if n_lines > 10:
    colors = tab60_colors(n_lines)
else:
    colors = tab10_colors(n_lines)
line_idx: int = 0
for i in range(m):
    for j in range(m):
        if j >= i:
            continue  # only plot windings once (lower triangle of w_target)
        ax_w_real.plot(
            time,
            w_real[:, i, j],
            linewidth=1.2,
            color=colors[line_idx],
            # label=f"w_{{{i}{j}}}",
            zorder=2,
        )
        line_idx += 1
ax_w_real.set_xlim(0, time[-1])
ax_w_real.set_ylim(w_real.min() - 0.5, w_real.max() + 0.5)
ax_w_real.grid(
    True,
    which="major",
    linestyle=":",
    color="gray",
    linewidth=0.5,
    zorder=1,
)
ax_w_real.grid(
    True,
    which="minor",
    linestyle=":",
    color="gray",
    linewidth=0.3,
    zorder=1,
)
ax_w_real.minorticks_on()
ax_w_real.set_xlabel(r"$t$ [s]")
ax_w_real.set_ylabel(r"$\bar{w}_{i,j}$")
ax_w_real.grid(True)


# 3. Tau
figsize = np.array([4, 4])  # width, height in cm
fig_tau: plt.Figure
ax_tau: plt.Axes
fig_tau, ax_tau = plt.subplots(figsize=figsize / 2.54)
tau: np.ndarray = data["tau_mat"]
tau_i: np.ndarray = data["tau_i_mat"]
time: np.ndarray = data["time"]
n, m = tau_i.shape
if m > 10:
    colors = tab60_colors(m)
else:
    colors = tab10_colors(m)
for i in range(m):
    ax_tau.plot(
        time,
        tau_i[:, i],
        linewidth=1.2,
        color=colors[i],
        zorder=2,
    )
ax_tau.plot(
    time,
    tau,
    linewidth=1,
    linestyle="--",
    color="k",
    zorder=3,
)
ax_tau.set_xlim(0, time[-1])
ax_tau.set_ylim(-0.1, 1.1)
ax_tau.grid(
    True,
    which="major",
    linestyle=":",
    color="gray",
    linewidth=0.5,
    zorder=1,
)
ax_tau.grid(
    True,
    which="minor",
    linestyle=":",
    color="gray",
    linewidth=0.3,
    zorder=1,
)
ax_tau.minorticks_on()
ax_tau.set_xlabel(r"$t$ [s]")
ax_tau.set_ylabel(r"$\tau(k), \tau_i(k)$")
ax_tau.grid(True)


# 4. Trajectories plot
figsize = np.array([6, 5])
fig_paths: plt.Figure = plt.figure(figsize=figsize / 2.54)
ax_paths: Axes3D = fig_paths.add_subplot(projection="3d", computed_zorder=False)
ax_paths.tick_params(axis="z", labelsize=6)
trajectories_mat: np.ndarray = data["trajectories"]
time: np.ndarray = data["time"]
x_lims: np.ndarray = data["x_lim"]
y_lims: np.ndarray = data["y_lim"]
ax_paths.set_xlim(x_lims)
ax_paths.set_ylim(y_lims)
ax_paths.set_zlim(0, time[-1])
ax_paths.set_proj_type("ortho")
pov = [45, 80, 0]
try:
    ax_paths.set_aspect("equalxy")
except (ValueError, NotImplementedError):
    xr = x_lims[1] - x_lims[0]
    yr = y_lims[1] - y_lims[0]
    ax_paths.set_box_aspect((xr, yr, 1.5))  # alternative: ax.set_box_aspect([1, 1, 2])
try:
    ax_paths.view_init(elev=pov[0], azim=pov[1], roll=pov[2])  # matplotlib >= 3.6
except TypeError:
    ax_paths.view_init(elev=pov[0], azim=pov[1])
ax_paths.set_xlabel("x")
ax_paths.set_ylabel("y")
ax_paths.set_zlabel("t")
n, _, m = trajectories_mat.shape
trajectories = []
for i in range(m):
    trajectories.append(trajectories_mat[:, :, i])
seg_points = []
seg_colors = []
for i, traj_i in enumerate(trajectories):
    traj_i = np.asarray(traj_i)
    for p1, p2 in zip(traj_i[:-1], traj_i[1:]):
        seg_points.append((p1, p2))  # endpoints of current segment ((x,y,z), (x,y,z))
        seg_colors.append(colors[i])  # color of current segment (depending on robot)
seg_points = np.array(seg_points)  # stack of endoints of each path segment (m*n, 2, 3)
mids = seg_points.mean(axis=1)  # midpoints of each segment (m*n, 3)
proj = ax_paths.get_proj()  # get projection direction based on POV
_, _, z_projected = proj3d.proj_transform(
    mids[:, 0],
    mids[:, 1],
    mids[:, 2],
    proj,
)  # project in proj coordinates, z_projected encodes projection depth
order = np.argsort(z_projected)
for rank, idx in enumerate(order):
    p1, p2 = seg_points[idx]
    ax_paths.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        color=seg_colors[idx],
        linewidth=1.2,
        zorder=rank,
    )

########################################################################################
# Extract braid from paths
braidlab = Braidlab()
braid, _ = braidlab.paths2braid(paths=trajectories_mat, angle=0)
print(f"braid word: {braid}")

########################################################################################
# Save figure
FilePath(folder / "images").mkdir(exist_ok=True)
fig_w_target.savefig(folder / "images/d_mpc_w_target.pdf", bbox_inches="tight")
fig_w_target.savefig(folder / "images/d_mpc_w_target.png", dpi=900, bbox_inches="tight")
fig_w_real.savefig(folder / "images/d_mpc_w_real.pdf", bbox_inches="tight")
fig_w_real.savefig(folder / "images/d_mpc_w_real.png", dpi=900, bbox_inches="tight")
fig_tau.savefig(folder / "images/d_mpc_tau.pdf", bbox_inches="tight")
fig_tau.savefig(folder / "images/d_mpc_tau.png", dpi=900, bbox_inches="tight")
fig_paths.savefig(folder / "images/d_mpc_path.pdf", bbox_inches="tight")
fig_paths.savefig(folder / "images/d_mpc_path.png", dpi=900, bbox_inches="tight")

# Show figures
# plt.show()
plt.close("all")
