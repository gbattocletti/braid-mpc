# Utils

Utility functions used in the projecct.


## Requirements

The MATLAB functions require braidlab to be installed and are executed from python through the MATLAB engine. The methods accessing braidlab are intended to be accessed through the python Braidlab class in the `braidlab.py` file.

The MATLAB functions are currently available only on Linux. The matlab engine can be installed via the optional dependeincy group `matlab` with
```sh
uv sync --extra matlab
```
The scripts were tested with braidlab release 3.2.6 with precompiled MEX files. 