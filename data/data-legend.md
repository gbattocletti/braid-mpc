# Data

## Filenames
Data filenames are composed by three main elements:
- a string indicating the type of topological specification (grid, braid, ...)
- a numerical value `mX` indicating the number of agents `m` considered
- a numerical index to distinguish between files

## Data fields
- x_lims: limits of the 2D workspace
- y_lims: limits of the 2D workspace
- m: number of agents
- x_init: initial state of each agent (m x 2 array)
- x_goal: goal state of each agent (m x 2 array)
- an additional data field corresponding to the type of topological specification

## Notes grids
- grids: topological specification as a sequence of m x m permutation grids
- x_init and x_goal must be coherent with initial and final relative positions in the
    topological specifications.