%% Test of conjugacy for braids with non-coincident endpoints
% Case with 5 agents where the initial and final positions of the agents are not the 
% same. In such scenario the initial and final conjugacy braids do not coincide. There 
% are two ways to compute the conjugate braids in this case:
% 1. compute two different initial and final braids corresponding to the permutations 
%    between the projections of the initial (and final, respectively) points on the two 
%    projection planes. The initial permutation must be 1->2, the final 2->1
% 2. use the braidlab closure operator on the paths to make so that the initial and 
%    final points coincide. Different options are available to disrupt as little as 
%    possible the resulting braids.

clear variables;
close all;
clc;

% Load packages
import braidlab.*

%% Define paths
n = 25;  % number of timesteps 
m = 5;  % number of agents
time = linspace(0, n-1, n);  % normalized time vector

% Space-time coordinates ([x, y] over time)
grids = zeros(n, m, m);
grids(1, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 2, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(2, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 2, 0, 0, 0];
            [0, 0, 5, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(3, :, :) = [
            [0, 0, 0, 0, 3];
            [4, 0, 0, 0, 0];
            [0, 2, 0, 0, 0];
            [0, 0, 5, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(4, :, :) = [
            [0, 0, 0, 3, 0];
            [4, 0, 0, 0, 0];
            [0, 2, 0, 0, 0];
            [0, 0, 5, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(5, :, :) = [
            [0, 0, 0, 3, 0];
            [0, 2, 0, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 0, 5, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(6, :, :) = [
            [0, 0, 0, 3, 0];
            [0, 0, 2, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(7, :, :) = [
            [0, 0, 3, 0, 0];
            [0, 0, 0, 2, 0];
            [4, 0, 0, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(8, :, :) = [
            [0, 0, 3, 0, 0];
            [0, 0, 0, 2, 0];
            [0, 4, 0, 0, 0];
            [5, 0, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(9, :, :) = [
            [0, 0, 0, 2, 0];
            [0, 0, 3, 0, 0];
            [0, 4, 0, 0, 0];
            [5, 0, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(10, :, :) = [
            [0, 0, 0, 2, 0];
            [0, 0, 3, 0, 0];
            [5, 0, 0, 0, 0];
            [0, 4, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(11, :, :) = [
            [0, 0, 0, 0, 2];
            [0, 0, 3, 0, 0];
            [5, 0, 0, 0, 0];
            [0, 4, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(12, :, :) = [
            [0, 0, 0, 0, 2];
            [0, 0, 3, 0, 0];
            [0, 5, 0, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(13, :, :) = [
            [0, 0, 0, 0, 2];
            [0, 0, 3, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(14, :, :) = [
            [0, 0, 0, 2, 0];
            [0, 0, 3, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(15, :, :) = [
            [0, 0, 0, 2, 0];
            [4, 0, 0, 0, 0];
            [0, 0, 3, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(16, :, :) = [
            [0, 0, 2, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 0, 0, 3, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 0, 1];
        ];
grids(17, :, :) = [
            [0, 0, 2, 0, 0];
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(18, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 2, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];
grids(19, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 2, 0, 0];
            [0, 5, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ];  % note: this corresponds to initial position of the agents
grids(20, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 5, 0, 0, 0];
            [0, 0, 2, 0, 0];
            [0, 0, 0, 1, 0];
        ]; 
grids(21, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 5, 0, 0];
            [0, 2, 0, 0, 0];
            [0, 0, 0, 1, 0];
        ]; 
grids(22, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 0, 5, 0];
            [0, 2, 0, 0, 0];
            [0, 0, 1, 0, 0];
        ]; 
grids(23, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 0, 5, 0];
            [0, 0, 1, 0, 0];
            [0, 2, 0, 0, 0];
        ]; 
grids(24, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 0, 5, 0];
            [0, 1, 0, 0, 0];
            [0, 0, 2, 0, 0];
        ];
grids(25, :, :) = [
            [4, 0, 0, 0, 0];
            [0, 0, 0, 0, 3];
            [0, 0, 0, 5, 0];
            [0, 0, 2, 0, 0];
            [0, 1, 0, 0, 0];
        ];

paths = grids2paths(grids);
paths_cl = closure(paths);  % close braid (coincidence between initial and final points)

%% Project braids and test conjugacy
% Select projection planes
ax_1 = 0;
ax_2 = pi/2;

% Compute projection of initial and final points to find a conjugacy braid
proj_initial_1 = project_points(squeeze(paths(1, :, :))', ax_1)';
proj_initial_2 = project_points(squeeze(paths(1, :, :))', ax_2)';
proj_final_1 = project_points(squeeze(paths(end, :, :))', ax_1)';
proj_final_2 = project_points(squeeze(paths(end, :, :))', ax_2)';

% Project braids
braid_1 = compact(braid(paths, ax_1));
braid_2 = compact(braid(paths, ax_2));

% Test standard conjugacy
[~, conj_braid] = conjtest(braid_1, braid_2);

% Manually input conjugacy braids 
% NOTE: initial and final conjugacy braids are different as starting and final points 
% do not match.
conj_braid_start = braid([3 2 3 4]);
conj_braid_end = braid([-4 -3 -2 -4 -3 -4]);
are_conjugate = conj_braid_start * braid_2 * conj_braid_end == braid_1;

% Print relevant info
fprintf('Braid 1: %s\n', char(braid_1));
fprintf('Braid 2: %s\n', char(braid_2));
% fprintf('Initial projection 1:');
% disp(proj_initial_1)
% fprintf('Initial projection 2:');
% disp(proj_initial_2)
% fprintf('Final projection 1:');
% disp(proj_final_1)
% fprintf('Final projection 2:');
% disp(proj_final_2)
fprintf('Conjugation braid initial: %s\n', char(compact(conj_braid_start)));
fprintf('Conjugation braid final: %s\n', char(compact(conj_braid_end)));
LogicalStr = {'false', 'true'};
fprintf('Conjugation succesfull: %s\n', LogicalStr{are_conjugate + 1});

%% Plot paths and braids
figure();
    subplot(131);
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
    subplot(132);
        plot(braid_1)
        grid 
        grid minor
        xlabel('x')
        ylabel('t')
    subplot(133);
        plot(braid_2)
        grid 
        grid minor
        xlabel('x')
        ylabel('t')