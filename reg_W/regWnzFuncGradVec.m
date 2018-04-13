function [J, gradW] = regWnzFuncGradVec(vecW, k, lambda1, lambda2, s1, s2)

if nargin<3, lambda1 = 1; end;
if nargin<4, lambda2 = lmabda1; end
if nargin<5, s1 = 0.01; end
if nargin<6, s2 = 10*s1; end 

d = length(vecW)/k;
W = reshape(vecW, d, k); % unroll variables for computations

%% Auxiliary constants for gradient/regularizer
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Cost function
J = lambda1*regW_fixed(W, k, s1, kE_term) + lambda2*regW_nz(W, k, s2);

%% Gradient
gradW = lambda1*gradW_opt_1_fixed(W, k, s1, Ik, E, CRt) + lambda2*gradW_nz(W, k, s2, Ik, R);