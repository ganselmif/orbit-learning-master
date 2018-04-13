% Wrapper function for setting up fminunc with multiple initial solutions
%
% costFunc: a handle to the cost function (and gradient)
% p: initial (and output) variable/parameters (vectorized)


function [p, manymins] = fminMsWrapper(costFunc, p, numstartpoints, startPoints, useParallel, MAX_ITER, optDisplay, tol)

if nargin<8, tol = 1e-8; end;
if nargin<7, optDisplay = 'off'; end;
if nargin<6, MAX_ITER = 100; end
if nargin<5, useParallel = true; end
if nargin<4, startPoints = 'randb'; end
if nargin<3, numstartpoints = 40; end

if strcmp(optDisplay, 'on')
    optDisplay = 'iter-detailed';
end

% BFGS ('bfgs') or Quasi-Newton ('dfp')
options = optimoptions('fminunc',...
    'MaxIter', MAX_ITER, ...
    'Algorithm', 'quasi-newton', ...
    'GradObj', 'on', ...
    'Diagnostics', 'off',...
    'HessUpdate', 'bfgs', ... % 'dfp'
    'Display', optDisplay, ...
    'TolFun', tol, ...
    'TolX', tol, ...
    'PlotFcns', @optimplotfval);

% Problem options
optprob = createOptimProblem('fminunc', 'objective', costFunc, 'x0', p, 'options', options);

% Multistart options
ms = MultiStart('UseParallel', useParallel, ...
    'StartPointsToRun', 'all', ... % 'PlotFcns', '@gsplotbestf', ...
    'Display', 'iter');

switch startPoints
    case 'randb'
        % Random initial points: need to define the bound!!
        startpts = RandomStartPointSet('ArtificialBound', 1, 'NumStartPoints', numstartpoints);
    case 'regu'
        % generate points in regular grid/project in unit ball
        % ok for small groups, scales extremely bad!
        [W, ~] = ndgridmat(length(p), numstartpoints);
        startpts = CustomStartPointSet(W');
    case 'uball'
        % Uniformly in unit circle
        W = randSampleVec(length(p), numstartpoints, 'uball');
        startpts = CustomStartPointSet(W');
    otherwise
        startpts = RandomStartPointSet('NumStartPoints', numstartpoints);
end

if useParallel && isempty(gcp), parpool; end

[p, ~, ~, ~, manymins] = run(ms, optprob, startpts);
% delete(gcp);

% startpts.list(optprob) % get access to sample points 
