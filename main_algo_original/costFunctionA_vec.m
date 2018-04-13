% Cost function with respect to A (basic sparse coding function) 
%
% vecA is a vectorized A
% J is a float
%
% NOT USED!

function J = costFunctionA_vec(vecA, X, D, lambda)

tic;
k = size(D, 2);
n = size(X, 2);

A = reshape(vecA, k, n); % unroll variables for computations
J = (0.5/n)*(sum(sum((X - D*A).^2))) + lambda*norm(vecA, 1);

% objFunc = @(p) (0.5/n)*(sum(sum((X - D*p).^2))) + lambda*norm(p(:), 1);
% J = objFunc(A)