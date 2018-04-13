% Regularizer and Gradient in a single function

function [J, gradW] = regWFuncGradVec(vecW, k, s)

if nargin<3, s = 0.1; end

d = length(vecW)/k;
W = reshape(vecW, d, k); % unroll variables for computations

%% Auxiliary constants for gradient/regularizer
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Cost function
J = regW_fixed(W, k, s, kE_term);

%% Gradient
gradW = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);