% Objective function of orbit learning problem
%
% lambdas = [lambda1; lambda2; lambda3; lambda4; lambda5];

function f = orbLearnObjectiveFunction(X, D, A, w, M, lambdas)

k = size(D, 2);
DtD = D'*D;
Jk = ones(k, k);
n = size(X,2);

f = (0.5/n)*norm(X - D*A, 'fro')^2 + ...
    0.5*lambdas(1)*norm(sum(Dw1(w, M, k), 3) - DtD, 'fro')^2 + ...
    0.5*lambdas(2)*termPermuM(M) + ...
    0.5*lambdas(3)*norm(sum(M, 3) -Jk, 'fro')^2 + ...
    0.5*lambdas(4)*termOrthoM(M) +... 
    lambdas(4)*sum(abs(M(:))) + ... % sum(sum(sum(abs(M)))
    lambdas(5)*sum(abs(A(:)));