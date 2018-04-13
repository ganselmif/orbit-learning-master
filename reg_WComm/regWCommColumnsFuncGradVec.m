% regWCommFuncGradVec.m + commuting with each single column in W
%
% See also: regWCommFuncGradVec.m regWCommReLUFuncGradVec.m regWCommDetCovFuncGradVec.m 

% TO-DO: Merge with regWCommFuncGradVec.m

function [J, gradW] = regWCommColumnsFuncGradVec(vecW, XXt, k, d, s, lambda)

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

% loop over columns
Jq = 0; grad_Wq = 0;
for q = 1:k
    R = Ik(:,q)*(Ik(:,q))';
    grad_Wq = grad_Wq + 4*comm(comm(W*R*W', XXt), XXt)*W*R;
    Jq = Jq + trace(comm(XXt, W*R*W')^2);
end
Jq = Jq/k; grad_Wq = grad_Wq/k; % average contributions, not sum

%% Cost function
J = regW_fixed(W, k, s, kE_term) - lambda*(trace(comm(XXt, WWt)^2) + Jq);

%% Gradient
grad_W1 = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);

% grad_W2 = 4*comm(comm(WWt, XXt), XXt)*W + grad_Wq;
grad_W2 = vec(4*comm2(WWt, XXt)*W/k + grad_Wq); % maybe a bit faster

gradW = grad_W1 + lambda*grad_W2;

