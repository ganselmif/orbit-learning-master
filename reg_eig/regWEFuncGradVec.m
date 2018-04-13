function [J, gradW] = regWEFuncGradVec(vecW, XXt, k, d, s, lambda)

if nargin<6, lambda = 1; end
if nargin<5, s = 0.1; end
% d = length(vecW)/k;
% XXt = X*X'; % cached/pre-computed

%% Cached computations
Ik = sparse(eye(k));
E = kron(Ik, ones(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Unroll variables
W = reshape(vecW, d, k);

% W = project_unit_norm(W);
% W = project_unit_norm(project_pos(W));
% W = project_unit(W);

%% Cost function
se = s; %0.1*s;
J = regW_fixed(W, k, s, kE_term) + lambda*regE(W, XXt, d, k, se);

%% Gradient
grad_W1 = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);
grad_W2 = vec(grad_XXT_eigenvec(W, XXt, d, k, se));

% Relative weighting of terms
gradW = grad_W1 + lambda*grad_W2;