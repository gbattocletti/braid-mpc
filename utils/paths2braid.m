function [braid, word] = paths2braid(paths, angle)
    % Extracts the braid corresponding to a set of path projected on a plane.
    braid = compact(braidlab.braid(paths, angle));
    word = braid.word;
end