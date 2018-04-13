% regWdist.m Orbit/permutation distance
%
% Computes the value of reg() for 2 vectors to check if one is a permutation 
% of the other. M is a (d x 2) matrix (i.e., 2 vectors arranged columnwise)

% From: regW1matrix_fixed.m

function distReg = regWdist(M, s, kE_term)

% M = gramMat(M); % linear feature map Gramian

[d, k] = size(M);
vecM = vec(M);
difG = bsxfun(@minus, vecM', vecM);
kd = k*d; % number of comparison elements

if nargin<3
    E = kron(eye(k), ones(d));
    % (kE-1): the additional term removes the double contribution of the diagonal elements
    kE_term = (k*E - 1) - 0.5*(k-1)*eye(kd);
end

reg_c = tril(kE_term.*exp(-((difG).^2)./s));

c = kd*(kd+1)/2; % scaling constant = number of lower triangular elements
distReg = sum(sum(reg_c))./c;