% Wrapper function for setting up contrained optimization solvers.
%
% costFunc: a handle to the cost function (and gradient)
% p: initial (and output) variable/parameters (vectorized)
% conType: 'nonlcon',  ...
% conFun: non-linear constraint function

function p = fminConWrapper(costFunc, p, numstartpoints, startPoints, useParallel, MAX_ITER, optDisplay, condOption)

tol = 1e-8; % tolerance for function, var, constraint

% if nargin<9, conType = 'nonlcon'; end

if nargin<7, optDisplay = 'off'; end;
if nargin<6, MAX_ITER = 100; end
% MultiStart options
if nargin<5, useParallel = true; end
if nargin<4, startPoints = 'randb'; end
if nargin<3, numstartpoints = 1; end


if strcmp(optDisplay, 'on')
    optDisplay = 'iter-detailed';
end

opts = optimoptions('fmincon',...
    'MaxIter', MAX_ITER, ...
    'Algorithm', 'interior-point', ... 'sqp', ...
    'GradObj', 'on', ...
    'MaxFunEvals', 10000, ...
    'TolFun', tol, 'TolX', tol, 'TolCon', tol, ...
    'Display', optDisplay, ...
    'PlotFcns', @optimplotfval);

% Explicit contraints
A = []; b = [];
Aeq = []; beq = [];
lb = []; ub = [];
conFun = [];

% switch conType
%     
%     case 'nonlcon' % Nonlinear constraints only (e.g., norm)        
%         conFun = isa(nonlcon, 'function_handle')
%         % conFun = @(t)unitball(t, k);
%     case 'lb'
%         lb = 
% end
        
% TODO: hacky way to input options, fix!
if isa(condOption,'function_handle')
    % Nonlinear constraints
    conFun = condOption; 
elseif isnumeric(condOption)
    % Positivity contraints
    lb = condOption;
elseif iscell(condOption);
    conFun = condOption{1}; 
    lb = condOption{2};    
end


if numstartpoints>1
    
    %% NOT WORKING ... need to test(?)
    
    % Problem options
    optprob = createOptimProblem('fmincon', 'objective', costFunc, ...
        'x0', p, 'Aineq', A, 'bineq', b, 'Aeq', Aeq, 'beq', beq, 'lb', lb, 'ub', ub, ...
        'nonlcon', conFun, 'options', opts);
    
    % Multistart options
    ms = MultiStart('UseParallel', useParallel, ...
        'StartPointsToRun', 'bounds-ineqs', ... % 'PlotFcns', '@gsplotbestf', ...
        'Display', 'iter');
    
    switch startPoints
        case 'randb'
            % Random initial points: need to define the bound!!
            startpts = RandomStartPointSet('ArtificialBound', 1, 'NumStartPoints', numstartpoints);
            %                 case 'regu'
            %                     % generate points in regular grid/project in unit ball
            %                     % ok for small groups, scales extremely bad!
            %                     [W, ~] = ndgridmat(length(p), numstartpoints);
            %                     startpts = CustomStartPointSet(W');
        otherwise
            startpts = RandomStartPointSet('NumStartPoints', numstartpoints);
    end
    
    if useParallel && isempty(gcp), parpool; end
    
    [p, fval, eflag, output, manymins] = run(ms, optprob, startpts);
    % delete(gcp);
else
    
    p = fmincon(costFunc, p, A, b, Aeq, beq, lb, ub, conFun, opts);
end






