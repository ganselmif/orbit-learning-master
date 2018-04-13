% Wrapper function for regdist applied with representation W on matrix X
%
% X: d x N
% W: d x k
% distMat: N x N
% dotProdMat: k x N
% s: the Gaussian std (default is 10)

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function [distMat, dotProdMat] = regWdistWrapper(W, X, s)
if nargin<3, s = 10; end

[d, N] = size(X);
% representation vector/map dimension
dimRepVec = size(W,2);

%% Representation: Normalized dot-product with dictionary

% dotProdMat = W'*Xv;

% zero-mean/unit-norm
% dotProdMat = normdotprod(W, Xv);

% unit norm dot-products
dotProdMat = project_unit_norm(W)'*project_unit_norm(X);

%% Distance function
% s = 10;
% Auxiliary
k2 = 2; E = kron(eye(k2), ones(dimRepVec)); kE_term = (k2*E - 1) - 0.5*(k2-1)*eye(dimRepVec*k2);
kd = 2*dimRepVec; % number of comparison elements
c = kd*(kd+1)/2; % scaling constant = number of lower triangular elements

%% Slow element-by-element version
% distFun = @(XI,XJ)(regWdist([XI, XJ], s, kE_term)); % @(Xi,Xj) distFun([Xi, Xj], s, kE_term)
%
% tic
% D = zeros(N);
% % apply column-wise
% for i=1:N-1
%     for j=i+1:N
%         D(i,j) = distFun(dotProdMat(:,i), dotProdMat(:,j));
%     end
% end
% Ds = D + D';
% t1 = toc;

%% Vectorized use with pdist2
distFun = @(XI,XJ)(regWdist1(XI, XJ, s, kE_term, c));
D = zeros(N);
for i=1:N
    % disp(i);
    D(i,i+1:end) = pdist2(dotProdMat(:,i)', dotProdMat(:,i+1:end)', distFun);
end
Ds = D + D';

%% Fully vectorized -- seems slower than the line-by-line version above
% distFun = @(XI,XJ)(regWdist2(XI, XJ, s, kE_term, c));
% tic; DD = pdist2(dotProdMat', dotProdMat', distFun); t3 = toc;

distMat = abs(Ds); % log(Ds);