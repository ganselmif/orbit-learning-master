% Wrapper function for setting up different optimization pipelines
%
% costFunc: a handle to the cost function (and gradient)
% p: initial (and output) variable/parameters (vectorized)
% optimType: 'fminunc', 'minFunc', ...

function p = fminWrapper(costFunc, p, optimType, MAX_ITER, optDisplay, tol)

if nargin<6, tol = 1e-8; end
if nargin<5, optDisplay = 'off'; end;
if nargin<4, MAX_ITER = 4; end

switch optimType
    
    case 'minFunc' % minFunc.m
        
        if strcmp(optDisplay, 'on')
            optDisplay = 'iter';
        end
        
        options.Method = 'lbfgs'; % L-BFGS (default)
        options.MaxIter = MAX_ITER;
        options.Display = optDisplay;
        options.optTol = tol;
        options.progTol = tol;
        p = minFuncMod(costFunc, p, options);
        
    otherwise % case 'fminunc' % MATLAB/Optimization Toolbox
        
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
            'TolFun', tol, ...     % FunctionTolerance, OptimalityTolerance
            'TolX', tol, ...       % StepTolerance
            'PlotFcns', @optimplotfval);
        
        [p, ~, ~, ~] = fminunc(costFunc, p, options);
        
        
end