% regWdist1.m Orbit/permutation distance
%
% Computes reg() for 2 vectors (X, Y) to check if one is a permutation
% of the other.

% From: regWdist.m for use with pdist2

function distReg = regWdist1(X, Y, s, kE_term, c)

% M = gramMat(M); % linear feature map Gramian
vecM = [X';Y']; % vec(M);
difG = bsxfun(@minus, vecM', vecM);

if nargin<5
    d = size(X, 2);
    kd = 2*d; % number of comparison elements
    c = kd*(kd+1)/2; % scaling constant = number of lower triangular elements
end
if nargin<4
    E = kron(eye(k), ones(d));
    % (kE-1): the additional term removes the double contribution of the diagonal elements
    kE_term = (k*E - 1) - 0.5*(k-1)*eye(kd);
end

reg_c = tril(kE_term.*exp(-((difG).^2)./s));
distReg = sum(sum(reg_c))./c;