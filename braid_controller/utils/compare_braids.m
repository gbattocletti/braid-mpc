function are_equal = compare_braids(word_1, word_2)
    % Check equality between two braid words
    braid_1 = compact(braidlab.braid(word_1));
    braid_2 = compact(braidlab.braid(word_2));
    if braid_1 == braid_2
        are_equal = true;
    else
        are_equal = false;
    end
end