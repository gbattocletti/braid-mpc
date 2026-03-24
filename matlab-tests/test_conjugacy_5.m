%% Test braid conjugacy
% 5 agents with looser permutation rules

clear variables;
close all;
clc;

% Load packages
import braidlab.*

%% Initialize grid sequence
% NOTE: the grid sequence currently allows for:
% - multiple swaps per timestep
% - jumps of more than 1 cell at a time (up to 2 cells in a single direction)

n = 10;  % timesteps
m = 5;  % agents

grids = zeros(n, m, m);

grids(1, :, :) = [
        [4, 0, 0, 0, 0];
        [0, 0, 0, 0, 3];
        [0, 0, 2, 0, 0];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 1, 0];
    ];
grids(2, :, :) = [
        [0, 0, 0, 0, 3];
        [4, 0, 0, 0, 0];
        [0, 2, 0, 0, 0];
        [0, 0, 5, 0, 0];
        [0, 0, 0, 1, 0];
    ];
grids(3, :, :) = [
        [0, 0, 0, 3, 0];
        [0, 0, 2, 0, 0];
        [4, 0, 0, 0, 0];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 0, 1];
    ];
grids(4, :, :) = [
        [0, 0, 3, 0, 0];
        [0, 0, 0, 2, 0];
        [0, 4, 0, 0, 0];
        [5, 0, 0, 0, 0];
        [0, 0, 0, 0, 1];
    ];
grids(5, :, :) = [
        [0, 0, 0, 2, 0];
        [0, 0, 3, 0, 0];
        [5, 0, 0, 0, 0];
        [0, 4, 0, 0, 0];
        [0, 0, 0, 0, 1];
    ];
grids(6, :, :) = [
        [0, 0, 0, 0, 2];
        [0, 0, 3, 0, 0];
        [0, 5, 0, 0, 0];
        [4, 0, 0, 0, 0];
        [0, 0, 0, 1, 0];
    ];
grids(7, :, :) = [
        [0, 0, 0, 2, 0];
        [0, 0, 3, 0, 0];
        [4, 0, 0, 0, 0];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 0, 1];
    ];
grids(8, :, :) = [
        [0, 0, 2, 0, 0];
        [4, 0, 0, 0, 0];
        [0, 0, 0, 3, 0];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 0, 1];
    ];
grids(9, :, :) = [
        [4, 0, 0, 0, 0];
        [0, 0, 2, 0, 0];
        [0, 0, 0, 0, 3];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 1, 0];
    ];
grids(10, :, :) = [
        [4, 0, 0, 0, 0];
        [0, 0, 0, 0, 3];
        [0, 0, 2, 0, 0];
        [0, 5, 0, 0, 0];
        [0, 0, 0, 1, 0];
    ];

%% Convert to paths
paths = grids2paths(grids);
time = 0:n-1;

% Plot paths
figure;
    hold on
    for agent = 1:m
        plot3(paths(:, 1, agent), paths(:, 2, agent), time, 'LineWidth', 1.3)
    end
    grid 
    grid minor
    xlim([1, m])
    ylim([1, m])
    zlim([0, n-1])
    xlabel('x')
    ylabel('y')
    zlabel('t')

view(45, 30)

%% Analyze braids

% Define projection axes
ax_1 = 0;
ax_2 = pi/3;

% Initial and final projections (to explore connection with conjugacy braid)
proj_init_1 = project_points(squeeze(paths(1, :, :))', ax_1)';
proj_init_2 = project_points(squeeze(paths(1, :, :))', ax_2)';
proj_end_1 = project_points(squeeze(paths(end, :, :))', ax_1)';
proj_end_2 = project_points(squeeze(paths(end, :, :))', ax_2)';

% Project braids
braid_1 = compact(braid(paths, ax_1));
braid_2 = compact(braid(paths, ax_2));

% Test conjugacy
[conj_test_12, conj_12] = conjtest(braid_1, braid_2);

% Print results
fprintf('Braid 1: %s\n', char(braid_1));
fprintf('Braid 2: %s\n', char(braid_2));
fprintf('Conjugation braid: %s\n', char(conj_12));

