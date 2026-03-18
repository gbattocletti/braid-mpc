%% Test braid conjugacy

clear variables;
close all;
clc;

% Load packages
import braidlab.*

%% Define paths
n = 6;  % number of timesteps 
m = 3;  % number of agents
time = linspace(0, n-1, n);  % normalized time vector

% Space-time coordinates ([x, y] over time)
X = zeros(n, 2, m);
X(:, :, 1) = [
                2, 3;
                2, 3;
                4, 3;
                4, 4;
                2, 4;
                2, 3;
            ];
X(:, :, 2) = [
                8, 2;
                8, 4;
                5, 5;
                3, 3;
                5, 5;
                8, 2;
             ];
X(:, :, 3) = [
                7, 6;
                6, 2;
                6, 2;
                6, 2;
                8, 5.1;
                7, 6;
             ];

%% Plot paths
figure;
    hold on
    for agent = 1:m
        plot3(X(:, 1, agent), X(:, 2, agent), time, 'LineWidth', 1.3)
    end
    grid 
    grid minor
    xlim([0, 10])
    ylim([0, 10])
    zlim([0, n])
    xlabel('x')
    ylabel('y')
    zlabel('t')

view(45, 30)

%% Project braids and test conjugacy
% Select projection plane
ax_1 = 0;
ax_2 = -pi/2;
ax_3 = pi/2;

% Project initial and final points
proj_init_1 = project_points(squeeze(X(1, :, :))', ax_1)';
proj_init_2 = project_points(squeeze(X(1, :, :))', ax_2)';
proj_init_3 = project_points(squeeze(X(1, :, :))', ax_3)';
proj_end_1 = project_points(squeeze(X(end, :, :))', ax_1)';
proj_end_2 = project_points(squeeze(X(end, :, :))', ax_2)';
proj_end_3 = project_points(squeeze(X(end, :, :))', ax_3)';

% Project braids
b_1 = compact(braid(X, ax_1));
b_2 = compact(braid(X, ax_2));
b_3 = compact(braid(X, ax_3));

% Test conjugacy
[conj_12, conj_braid_12] = conjtest(b_1, b_2);
[conj_13, conj_braid_13] = conjtest(b_1, b_3);
[conj_23, conj_braid_23] = conjtest(b_2, b_3);

% Print results
disp(b_1)
disp(b_2)
disp(b_3)
disp(compact(conj_braid_12))
disp(compact(conj_braid_13))
disp(compact(conj_braid_23))
