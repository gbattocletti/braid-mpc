function sorted_indices = project_points(points, angle_rad)
% PROJECT_POINTS Projects 2D points onto an axis and sorts them by distance 
% from origin.
%
% Inputs:
%   points     - Nx2 matrix of 2D points, where each row is [x, y]
%   angle_rad  - angle of the projection axis in radians (measured from x-axis)
%
% Output:
%   sorted_indices - indices of points ordered from closest to farthest
%                    projected distance from 0 along the axis

    % Unit vector along the projection axis
    axis_vec = [cos(angle_rad), sin(angle_rad)];

    % Project each point onto the axis via dot product
    % --> projections is an Nx1 vector of signed scalar projections
    projections = points * axis_vec';

    % Sort by absolute value (distance from origin along the axis)
    [~, sorted_indices] = sort(abs(projections));

end