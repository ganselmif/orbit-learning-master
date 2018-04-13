% Update of the w vector
%
% lambda1: regularization parameter of the
%
% Dependencies (either of the following):
% minFunc.m (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
% fminunc.m (MATLAB Optimization Toolbox)

function w = blockwL_f(MAX_ITER, M, D, w, lambda1, optimType)

if nargin<6, optimType = 'fminunc'; end

costFunc = @(p) costFunctionW_vec(p, M, D, lambda1);

w = fminWrapper(costFunc, w, optimType, MAX_ITER);

w  = project_unit(w); % w = w./norm(w);