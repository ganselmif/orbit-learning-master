% Joint regularizer and gradient computation for regCN
%
% Input is a vector
% output is a value (J) + a vector (gradW)

function [J, gradW] = regCNFuncGradVec(vecW, maxp, k, d, s, lambda)

if nargin<6, lambda = 1; end
if nargin<5, s = 0.1; end
if nargin<4, k = numel(vecW)/d; end

W = reshape(vecW, d, k); % unroll variables for computations

%% Cached computations
Jk = ones(k);
Jd = ones(d);
Id = eye(k);
k_term = k*Id-Jk;

E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';


R = 0;
gradCN = zeros([d, k]);

for p = 1:maxp
    
    Wp = W.^p;
    %Wp = exp(p*W);
    N = trace(Wp'*Wp);
    
    Wpm1 = W.^(p-1);
    %Wpm1 = exp(p*W);
    
    %% Regularizer
    R = R + (2/N)*trace(Wp*(k_term)*Wp'*Jd);  
     
    %% Gradient
    %dWCN = project_unit(dWCN + i*((2/N)*(Jd*Wi*2*(k*Id-Jk))-(2/N^2)*trace(Wi*2*(k*Id-Jk)*Wi'*Jd)*Wi).*Wim1);
    gradCN = gradCN + p*((2/N)*(Jd*Wp*2*k_term)-(2/N^2)*trace(Wp*2*k_term*Wp'*Jd)*Wp).*Wpm1;
end


%% Total cost function
J = regW_fixed(W, k, s, kE_term) + lambda*R;

%% Gradient
grad_W1 = vec(gradCN);
grad_W2 = gradW_opt_1_fixed(W, k, s, Ik, E, CRt);

gradW = lambda*grad_W1 + grad_W2;
