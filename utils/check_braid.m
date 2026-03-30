function is_correct = check_braid(paths, braid_target, angle)
    % CHECK_BRAID Checks if the paths give rise to a braid corresponding to the input
    %   braid when projected on the plane corresponding to the angle.
    %
    %   is_correct = CHECK_BRAID(paths, braid, angle) verifies whether the given paths
    %   produce a braid that matches the specified braid.
    %
    %   Inputs:
    %       paths - Paths to evaluate
    %       braid_target - Target braid to match against
    %       angle - Angle specifying the projection plane to generate the braid
    %
    %   Output:
    %       is_correct - Boolean indicating whether paths correspond to the braid

    % Extract the braid from the paths
    braid = compact(braidlab.braid(paths, angle));

    % Check if the extracted braid matches the target braid
    is_correct = isequal(braid, compact(braid_target));
end
