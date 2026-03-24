import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from data import grids_2
from utils import braidlab, invariants
from visualization import plot

# Load data
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

# Compute braid representations (only supported on linux)
if sys.platform.startswith("linux") and shutil.which("matlab") is not None:
    b = braidlab.Braidlab()
    braid, braid_matlab = b.paths2braid(paths, angle=0 * np.pi / 180)
    b.plot_braid(braid_matlab)
    print("Braid representation:", braid)

# Show all plots
plt.show()
