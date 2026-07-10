function [braid, word] = paths2braid(paths, angle, flip_generators)
    % Extracts the braid corresponding to a set of path projected on a plane.
    % paths = braidlab.closure(paths);
    warning('off', 'BRAIDLAB:braid:colorbraiding:notclosed')
    braid = compact(braidlab.braid(paths, angle));
    word = braid.word;
    if flip_generators == true
        word = -word;
        braid = compact(braidlab.braid(word));
    end
end