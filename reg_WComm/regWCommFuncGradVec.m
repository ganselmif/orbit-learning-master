function [J, gradW] = regWCommFuncGradVec(vecW, XXt, k, d, s, lambda)

if nargin<6, lambda = 1; end
if nargin<5, s = 0.1; end
% d = length(vecW)/k;
% XXt = X*X'; % cached/pre-computed

%% Cached computations
Ik = eye(k);
E = kron(Ik, ones(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Unroll variables and covariance
W = reshape(vecW, d, k);
WWt = W*W'/k; % divide by number of elements

%% Cost function
J = regW_fixed(W, k, s, kE_term) - lambda*trace(comm(XXt, WWt)^2);

%% Gradient
grad_W1 = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);

% grad_W1 = 4*comm(comm(WWt, XXt), XXt)*W;
grad_W2 = vec(4*comm2(WWt, XXt)*W/k); % maybe a bit faster

gradW = grad_W1 + lambda*grad_W2;