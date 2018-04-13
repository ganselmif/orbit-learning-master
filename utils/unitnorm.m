% Constraint function for fmincon
% See: http://www.mathworks.com/help/optim/ug/fmincon.html#busow0u-1_1
%
% Unit norm contraint(s)
%  Inequality constraint: []  
%  Equality constraint: 
%
% See also fmincon.m unitball.m
 
function [c, ceq] = unitnorm(X, d, N)

c = []; % inequality contraints 

X = reshape(X, d, N); % unroll variables for computations
ceq = sqrt(sum(X.^2)) - ones(1, N); % linear equality (norm equal to 1)