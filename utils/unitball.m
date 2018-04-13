% Constraint function for fmincon
% See: http://www.mathworks.com/help/optim/ug/fmincon.html#busow0u-1_1
%
% Unit ball contraint(s)
%  Inequality constraint: \sum_i ||w||^2 < k 
%  Equality constraint: []
%
% See also fmincon.m
 
function [c, ceq] = unitball(X, k)

% X = reshape(X, length(X)/k, k); % unroll variables for computations
% X = bsxfun(@rdivide, X, max(sqrt(sum(X.^2)),1));

c = sum(sum(X.^2)) - k; % linear inequality
ceq = [];
