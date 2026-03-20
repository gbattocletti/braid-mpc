import matplotlib.pyplot as plt

from data import grids_2
from utils import invariants
from visualization import plot

# Load data
grids = grids_2.grids

# Convert grids to paths
paths = invariants.grids2paths(grids)
plot.plot_paths_3d(paths, show=False)

# Compute winding numbers
windings_linear = invariants.paths2windings(
    paths,
    upscale_factor=20,
    intermediate_shape="linear",
)
plot.plot_windings(windings_linear, show=False)
windings_spline = invariants.paths2windings(
    paths,
    upscale_factor=20,
    intermediate_shape="spline",
)
plot.plot_windings(windings_spline, show=False)

# Show all plots
plt.show()
