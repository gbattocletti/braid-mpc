function paths = grids2paths(grids)
    % GRID2PATH transforms an n*m*m sequence of grids into a n*2* m
    %   set of paths
    %
    % Inputs:
    %   grid    - an n*m*m matrix, where each m*m slice represent a grid of
    %             the relative positions of the robot. In each row and
    %             column there is only one nonzero element, corresponding
    %             to the index of a robot. The indexes are between 1 and m.
    %
    % Output:
    %   paths   - an n*2*m matrix of 2D points. Each point corresponds to
    %             the coordinates of one of the grid nodes. Points
    %             corresponding to the same agent through the different
    %             permutation grids are collected in n*2 column matrices.

    % Extract dimensions m and n
    s = size(grids);

    % Validate input dimensions
    if length(s) ~= 3
        error("The input matrix does not have the right dimension [n, m, m]")
    end
    if s(2) ~= s(3)
        error("The dimensions of the grids must be the same")
    end

    % Store matrix dimensions
    n = s(1);
    m = s(2);

    % Initialize paths
    paths = zeros(n, 2, m);

    % Iterate over grids and build paths
    for agent = 1:m
        for row = 1:n
            [x, y] = find(squeeze(grids(row, :, :)) == agent);
            if isempty(x) || isempty(y)
                error("Index %i not found at row %i", agent, row )
            elseif length(x) > 1 || length(y) > 1
                error("Duplicate index %i", agent)
            else
                paths(row, :, agent) = [x, y];
            end
        end
    end
end