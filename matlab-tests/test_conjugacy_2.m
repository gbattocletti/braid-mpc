%% Test braid conjugacy
% Case with 3 agents

clear variables;
close all;
clc;

% Load packages
import braidlab.*

%% Define paths
n = 13;  % number of timesteps 
m = 3;  % number of agents
time = linspace(0, n-1, n);  % normalized time vector

% Space-time coordinates ([x, y] over time)
grids = zeros(n, m, m);
grids(1, :, :) = [
            [0, 2, 0];
            [0, 0, 3];
            [1, 0, 0];
        ];
grids(2, :, :) = [
            [2, 0, 0];
            [0, 0, 3];
            [0, 1, 0];
        ];
grids(3, :, :) = [
            [2, 0, 0];
            [0, 3, 0];
            [0, 0, 1];
        ];
grids(4, :, :) = [
            [0, 3, 0];
            [2, 0, 0];
            [0, 0, 1];
        ];
grids(5, :, :) = [
            [0, 3, 0];
            [0, 0, 1];
            [2, 0, 0];
        ];
grids(6, :, :) = [
            [0, 0, 1];
            [0, 3, 0];
            [2, 0, 0];
        ];
grids(7, :, :) = [
            [0, 1, 0];
            [0, 0, 3];
            [2, 0, 0];
        ];
grids(8, :, :) = [
            [1, 0, 0];
            [0, 0, 3];
            [0, 2, 0];
        ];
grids(9, :, :) = [
            [1, 0, 0];
            [0, 3, 0];
            [0, 0, 2];
        ];
grids(10, :, :) = [
            [1, 0, 0];
            [0, 0, 2];
            [0, 3, 0];
        ];
grids(11, :, :) = [
            [0, 0, 2];
            [1, 0, 0];
            [0, 3, 0];
        ];
grids(12, :, :) = [
            [0, 0, 2];
            [0, 3, 0];
            [1, 0, 0];
        ];
grids(13, :, :) = [
            [0, 2, 0];
            [0, 0, 3];
            [1, 0, 0];
        ];

%% Plot paths
paths = grids2paths(grids);

figure;
    hold on
    for agent = 1:m
        plot3(paths(:, 1, agent), paths(:, 2, agent), time, 'LineWidth', 1.5)
    end
    daspect([1 1 1])
    pbaspect([1 1 1])
    grid 
    grid minor
    xlim([0, m+1])
    ylim([0, m+1])
    zlim([0, n])
    xlabel('x')
    ylabel('y')
    zlabel('t')

view(45, 30)

%% Project braids and test conjugacy
% Select projection plane
ax_1 = 0;
ax_2 = pi/3;
ax_3 = pi/2;

% Project initial and final points
proj_init_1 = project_points(squeeze(paths(1, :, :))', ax_1)';
proj_init_2 = project_points(squeeze(paths(1, :, :))', ax_2)';
proj_init_3 = project_points(squeeze(paths(1, :, :))', ax_3)';
proj_end_1 = project_points(squeeze(paths(end, :, :))', ax_1)';
proj_end_2 = project_points(squeeze(paths(end, :, :))', ax_2)';
proj_end_3 = project_points(squeeze(paths(end, :, :))', ax_3)';

% Project braids
b_1 = compact(braid(paths, ax_1));
b_2 = compact(braid(paths, ax_2));
b_3 = compact(braid(paths, ax_3));

% Test conjugacy
[conj_12, conj_braid_12] = conjtest(b_1, b_2);
[conj_13, conj_braid_13] = conjtest(b_1, b_3);
[conj_23, conj_braid_23] = conjtest(b_2, b_3);

% Print results
fprintf('Braid 1: %s\n', char(b_1));
fprintf('Braid 2: %s\n', char(b_2));
fprintf('Braid 3: %s\n', char(b_3));
fprintf('Conjugation braid 12: %s\n', char(compact(conj_braid_12)));
fprintf('Conjugation braid 23: %s\n', char(compact(conj_braid_23)));
fprintf('Conjugation braid 13: %s\n', char(compact(conj_braid_13)));
