import braidlab.*
clear variables;
close all;
clc;

% Define braid and strand colors
word = [-8  1  6 -2 -3  7  4  8 -5  3 -6  9  2  1 -4 -8  5  4  7  8  3];
% word = [1, 1, -2];
colors = {"#f00000", "#bb7000", "#3887ff", "#b92121", "#ffc814", ...
          "#0b3b7e", "#9c0000", "#ffb52c", "#0087a8", "#003877"};
linewidth = 1.5;

% Create figure
figure('Visible', 'off');
plot(braid(word));
strands = findobj(gca, 'Type', 'line');
for k = 1:numel(strands)
    strands(k).Color = colors{k};  % apply colors to black strands
    strands(k).LineWidth = linewidth;
end
axis off;

% Save figure
exportgraphics(gca, 'braid.pdf', 'ContentType', 'vector', ...
               'BackgroundColor', 'none');
close all;