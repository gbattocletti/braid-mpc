import braidlab.*
clear variables;
close all;
clc;

% Define braid and strand colors
word = [1, 1, -2];
colors = {"#a50000", "#ffa217", "#3887ff"};
linewidth = 1;

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