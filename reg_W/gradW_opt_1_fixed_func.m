% Gradient of reg(W) wrt. W, i.e. dreg(W)/dW 
% 
% Using (fixed) regularizer with generic (nonlinear) Gramian
%
%
% % Example (check gradient computation)
% k = 20; W = rand(k, k); % use random matrix
% testgradW(W, s, 6, 'relu');  % relu feature map
% testgradW(W, s, 6, 'pow', 2); % power 2 feature map
%
% See also: gramMat.m regW_fixed_func.m script_test_reg_grad.m

function dW = gradW_opt_1_fixed_func(W, k, s, Ik, E, CRt, varargin)

if nargin<4 || isempty(Ik)
    Ik = sparse(eye(k)); %Ik = eye(k);
    E = kron(Ik, ones(k));
end

% Gramian with feature map
[G, fW, dfdW] = gramMat(W, varargin{:});
vecG = vec(G);

% sign is important: has to have the opposite sign than the regularizer!?
difG = bsxfun(@minus, vecG', vecG);
% dev = tril(bsxfun(@minus, vecG, vecG'));
% clear vecG;

% derivative of exponential
dev = -(2/s).*(difG.*exp(-((difG).^2)./s));

%% (kE-1): the additional term removes the double contribution of the diagonal elements
% for the gradient here it is OK since we will reject the diagonal + upper triangular
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
dev = kE_term.*dev;

% tic; I = dev(itril(size(dev),-1)); toc
% same computation (& much faster) without calling itril
I = dev(~triu(ones(k^2)));
%clear E dev;

% R = sparse(eye(k^2) + genT_opt(k));
% C = genC_opt(k);
if nargin<6 || isempty(CRt)
    % create the auxiliary matrices
    [C, R] = gradW_opt_aux(k);
    CRt = R'*C';
end

% P = C*R; % this multiplication was killing it!
%clear R C;
c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
dW = vec(dfdW).*(kron(Ik, fW)*(CRt*I)/c); % parenthesis are needed for sparse gpuArrays

