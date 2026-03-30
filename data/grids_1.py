"""
Input data for simulations.

DATA:
    - m: number of agents
    - x_lims: limits of the 2D workspace
    - y_lims: limits of the 2D workspace
    - x_init: initial state of each agent (m x 2 array)
    - x_goal: goal state of each agent (m x 2 array)
    - grids: topological specification as a sequence of m x m permutation grids


NOTE:
    - multiple swaps per timestep are allowed
    - moves of more than one node per timestep are allowed
    - the initial and final positions of the agents in the permutation grids is the same
    - x_init and x_goal must be coherent with initial and final relative positions in
      the topological specifications.
"""

import numpy as np

# Simulation settings
m = 5  # number of agents
x_lims = [0, 10]
y_lims = [0, 10]

# Initial states
x_init = np.array(
    [
        [7.3, 1.0],
        [4.0, 5.0],
        [8.0, 8.0],
        [0.3, 8.8],
        [2.0, 2.0],
    ]
)

# Goal states
x_goal = np.array(
    [
        [5.5, 0.1],
        [3.5, 7.0],
        [8.0, 9.0],
        [1.2, 9.5],
        [2.0, 3.0],
    ]
)

# Topological specifications (sequence of permutationgrids)
grids = np.array(
    [
        [
            [4, 0, 0, 0, 0],
            [0, 0, 0, 0, 3],
            [0, 0, 2, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 0, 3],
            [4, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 5, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 3, 0],
            [0, 0, 2, 0, 0],
            [4, 0, 0, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 3, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [5, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [0, 5, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [4, 0, 0, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 2, 0, 0],
            [4, 0, 0, 0, 0],
            [0, 0, 0, 3, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [4, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 3],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        [
            [4, 0, 0, 0, 0],
            [0, 0, 0, 0, 3],
            [0, 0, 2, 0, 0],
            [0, 5, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    ]
)
