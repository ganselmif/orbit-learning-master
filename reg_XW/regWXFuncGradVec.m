function [J, gradW] = regWXFuncGradVec(vecW, X, k, d, s, lambda)

if nargin<6, lambda = 1; end
if nargin<5, s = 0.1; end
% d = length(vecW)/k;

W = reshape(vecW, d, k); % unroll variables for computations

%% Cached computations
Ik = sparse(eye(k));
E = kron(Ik, ones(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Cost function
J = regW_fixed(W, k, s, kE_term) + lambda*regW_fixed(X'*W, k, s, kE_term);

%% Gradient
grad_W1 = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);
grad_W2 = gradWX_opt_1_fixed(W, X, k, s, Ik, E, CRt);

gradW = grad_W1 + lambda*grad_W2;