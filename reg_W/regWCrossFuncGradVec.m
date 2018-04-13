function [J, gradW] = regWCrossFuncGradVec(vecW, groups, k, lambda1, lambda2, s1, s2)

if nargin<4, lambda1 = 1; end;
if nargin<5, lambda2 = lambda1; end
if nargin<6, s1 = 0.01; end
if nargin<7, s2 = 10*s1; end

K = numel(groups);
d = length(vecW)/K;
W = reshape(vecW, d, K); % unroll variables for computations

%% Cached computations
[C, R] = gradW_opt_aux(k);
T = sparse(genT_opt(k));
if isa(W, 'gpuArray');
    C = gpuArray(C);
    R = gpuArray(R);
    Ik = gpuArray(eye(k));
    % Ik = gpuArray.speye(k);
    T = gpuArray(T);
else
    C = sparse(C);
    Ik = sparse(eye(k)); %Ik = eye(k);
end
E = kron(Ik, ones(k));
Ct = C';
CRt = R'*C';
CTt = T*Ct;
varargin_grad = {s1, Ik, E, CRt, Ct, CTt};

%% Cost function
gradW2 = [];
for g=1:length(unique(groups))
    Wg = W(:, groups==g);
    regW2(g) = regW_nz(Wg, k, s2);
    gradW2 = [gradW2 reshape(gradW_nz(Wg, k, s2, Ik, R), d, k)]; 
end

J = lambda1*regW_mult_cross(W, groups, k, s1, E) + lambda2*sum(regW2);
gradW = lambda1*gradW_opt_1_mult_cross(W, groups, k, varargin_grad{:}) + lambda2*gradW2(:);


