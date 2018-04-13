% Generate a matrix (d x Nm) of points on the unit ball from the closest
% points on a d-dimensional grid.
%
% W = ndgrid(N, d) will compute the closest number of points Nm =
% round(N^1/d) to regularily sample the d-dim cube, perturb them by 0.01
% var and project them on the d-dim unit ball.
%
% Use: generate random seeds (regularily spaced) for MultiStart optimizers
% 
% See also ndgrid.m

% GE, LCSL/CBMM/MIT, 05/23/2016

function [W, nPoints] = ndgridmat(nDim, N)
% d = 6; N = 40;

nPointsInDim = round(N^(1/nDim)); % sample the cube
nPoints = nPointsInDim^nDim; % actual number of points: W will be d x nPoints

r = linspace(0, 1, nPointsInDim); % range in each dimension

% generic in/out arguments for ndgrid
str_range = repmat('r,', 1, nDim); str_range = str_range(1:end-1);
str_x = []; 
for d = 1:nDim
    str_x = [str_x sprintf('x%d,', d)];
end
str_x = str_x(1:end-1);

% call ndgrid.m with arbitrary num of inputs/outputs
eval(sprintf('[%s] = ndgrid(%s);', str_x, str_range))

% place into single matrix
W = [];
for d=1:nDim
    eval(sprintf('W = [W; x%d(:)''];', d))
end

% perturb integer points
W = W + 0.01*randn(size(W));

% reject points outside d-dimensional unit ball
% sqrt(sum(W.^2, 1))

% project inside unit ball
W = bsxfun(@rdivide, W, max(sqrt(sum(W.^2)),1));

% shuffle
W = W(:, randperm(nPoints));




