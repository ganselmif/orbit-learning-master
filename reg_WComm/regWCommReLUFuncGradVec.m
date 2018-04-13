% regWCommFuncGradVec.m + multiple reg(f(W)) terms (e.g. ReLUs)
%
% See also: regWCommFuncGradVec.m regWCommColumnsFuncGradVec.m regWCommDetCovFuncGradVec.m 

% TO-DO: Merge with regWCommFuncGradVec.m

function [J, gradW] = regWCommReLUFuncGradVec(vecW, XXt, k, d, s, lambda_w, lambda_n, lambda_s)

if nargin<8, lambda_s = 1; end
if nargin<7, lambda_n = 1; end
if nargin<6, lambda_w = 1; end
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

%% Cost function

% 1. Regularizer
Jw = regW_fixed(W, k, s, kE_term);
grad_W_w = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);

% 2. ReLU non-linear constraints
Jr = 0; grad_W_r = zeros(size(vecW));
if lambda_n~=0    
    num_thresh = 100;
    p_thresh = linspace(-0.6, 0.6, num_thresh);
    % Aggregate gradients and regularizer
    for p = p_thresh
        % arg_r = {'relu', p};
        Jr = Jr + regW_fixed_func(W, k, s, kE_term, 'relu', p);
        grad_W_r = grad_W_r + gradW_opt_1_fixed_func(W, k, s, Ik, E, CRt, 'relu', p);
    end
    lambda_n = lambda_n/num_thresh; % account for the number of ReLUs used
end

% 3. Commutator
[Jc, grad_W_c] = wComm(W, XXt, k);

% 4.  Self-coherence
if lambda_s~=0   
    [Js, grad_W_s] = wSelfCoherence(W, k);
else 
    Js = 0; grad_W_s = zeros(size(vecW));
end

%% Cost function 
J = Jw + lambda_w*Jc + lambda_n*Jr + lambda_s*Js;

%% Gradient (vectorized)
gradW = grad_W_w + lambda_w*vec(grad_W_c) + lambda_n*grad_W_r + lambda_s*vec(grad_W_s);