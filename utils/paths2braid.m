function [braid, word] = paths2braid(paths, angle)
    % PATHS2BRAID Extracts the braid corresponding to a set of paths projected on a 
    %   plane.
    %
    %   Inputs:
    %       paths - Paths to project
    %       angle - Angle specifying the projection plane to generate the braid
    %
    %   Output:
    %       braid - Extracted braid
    %       word - Word corresponding to the braid

    % Extract the braid from the paths
    braid = compact(braidlab.braid(paths, angle));

    % Extract the braid word
    word = braid.word;
end