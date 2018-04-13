% Optimized computation for gradW with additional inputs
% and corresponding to reg^2.
% 
% From gradW_opt, gradW_opt_1

function dW = gradW_opt_2(W, k, s, Ik, E, CRt)

if nargin<4
    Ik = sparse(eye(k)); %Ik = eye(k);
    E = kron(Ik, ones(k));
end

vecG = vec(W'*W);

% sign is important: has to have the opposite sign than the regularizer!?
dev = tril(bsxfun(@minus, vecG', vecG));
% dev = tril(bsxfun(@minus, vecG, vecG'));
% clear vecG;

% term is reused for both reg. and gradient: compute once
dev_reg = (k.*E - 1).*exp(-(dev.^2)./s);

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
% Regularizer
reg = sum(sum(dev_reg))/c;

% Gradient terms
dev = -(2/s)*dev.*dev_reg;

I = dev(~triu(ones(size(dev))));
%clear E dev;

% R = sparse(eye(k^2) + genT_opt(k));
% C = genC_opt(k);
if nargin<6
    % create the auxiliary matrices
    [C, R] = gradW_opt_aux(k);
    CRt = R'*C';
end

% P = C*R; % this multiplication was killing it!
%clear R C;
dW = 2*reg*kron(Ik, W)*(CRt*I)/c; % parenthesis are needed for sparse gpuArrays
% dW = 2*reg*sparse(kron(Ik, W))*CRt*I;







