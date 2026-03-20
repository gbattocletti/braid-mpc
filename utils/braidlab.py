"""
Utility functions to use the MATLAB engine and the braidlab library within it.
"""

import sys

import numpy as np

if sys.platform.startswith("linux"):
    import matlab
    import matlab.engine


class Braidlab:
    def __init__(self):
        if sys.platform.startswith("linux"):
            self.engine: matlab.engine.MatlabEngine = matlab.engine.start_matlab()
        else:
            raise NotImplementedError(
                "MATLAB functions are not supported on this platform due to the "
                "dependency on the braidlab library."
            )

    def paths2braid(self, paths: np.ndarray, angle: float) -> np.ndarray:
        """
        Convert paths to braids using the braidlab library in MATLAB.

        Args:
            paths (np.ndarray): A 3D array of shape (n, 2, m) representing m 2D paths of
                n steps of length. Each entry paths[t, :, i] gives the (x, y)
                coordinates of agent i at time step t.
            angle (float): The angle in degrees to use for the braid projection.

        Returns:
            np.ndarray: A 1D array of shape representing the braid.

        Raises:
            TypeError: If the input 'paths' is not a numpy array.
            TypeError: If the input 'angle' is not a float.
            ValueError: If the input 'paths' does not have the correct shape.
        """
        # Validate inputs
        if not isinstance(paths, np.ndarray):
            raise TypeError("Input 'paths' must be a numpy array.")
        if not isinstance(angle, (float, int)):
            raise TypeError("Input 'angle' must be a float or int.")
        if paths.ndim != 3:
            raise ValueError("Input 'paths' must be a 3D array of shape (n, 2, m).")
        if paths.shape[1] != 2:
            raise ValueError(
                "The second dimension of 'paths' must be 2 for (x, y) coordinates."
            )

        # Convert paths to MATLAB format
        matlab_paths = matlab.double(paths.tolist())
        matlab_angle = float(angle)
        matlab_angle = float(angle)

        # Call the MATLAB function to convert paths to braids
        # matlab_braid = self.engine.compact(
        #     self.engine.braidlab.braid(matlab_paths, matlab_angle)
        # )
        # braid = np.array(self.engine.eval("matlab_braid.word", nargout=1))
        self.engine.cd(r"utils", nargout=0)
        braid = np.array(
            self.engine.paths2braid(matlab_paths, matlab_angle, nargout=1)
        )[0]

        return braid
