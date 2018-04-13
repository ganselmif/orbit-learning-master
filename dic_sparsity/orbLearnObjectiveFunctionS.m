% Objective function of orbit learning problem (modified with reg(W) and
% sparsity only)

function f = orbLearnObjectiveFunctionS(X, D, A, lambdaA, lambdaD, k, s)

if nargin<7, s = 0.001; end
if nargin<6, k = size(D, 2); end

n = size(X,2);
f = (0.5/n)*norm(X - D*A, 'fro')^2 + lambdaA*norm(A, 1);

if exist('lambdaD', 'var') && lambdaD~=0
    f = f + lambdaD*regW_fixed(D, k, s);
end