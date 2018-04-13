% h = imagesc(W);
% impixelregion(h);

% Source : http://stackoverflow.com/questions/3942892/how-do-i-visualize-a-matrix-with-colors-and-values-displayed
function h = imagescpix(M, cmap, fn)

if nargin<3, fn = 10; end
if nargin==1; cmap = flipud(gray); end % inverce gray colormap (higher values are black and lower values are white)

h = figure; imagesc(M); colormap(cmap);

[nRows, mCols] = size(M);

textStrings = num2str(M(:),'%0.2f');          % create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % remove any space padding

[x,y] = meshgrid(1:mCols, 1:nRows);   % coordinates for strings
hStrings = text(x(:),y(:),textStrings(:),...      % superimpose strings
    'HorizontalAlignment','center', 'fontsize', fn);

% !!! TO-DO: NOT WORKING FOR non-gray colormaps. Not printing in white. 
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(M(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'},num2cell(textColors, 2));  %# Change the text colors

% set(gca,'XTick',1:5,...                         %# Change the axes tick marks
%        'XTickLabel',{'A','B','C','D','E'},...  %#   and tick labels
%        'YTick',1:5,...
%        'YTickLabel',{'A','B','C','D','E'},...
%        'TickLength',[0 0]);