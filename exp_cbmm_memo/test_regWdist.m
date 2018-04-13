% Compute the ordinary Euclidean distance
X = randn(100, 5);
Y = randn(25, 5);


% Use a function handle to compute a distance that weights each
% coordinate contribution differently.
Wgts = [.1 .3 .3 .2 .1];            % coordinate weights
weuc = @(XI,XJ)(sqrt(bsxfun(@minus,XI,XJ).^2));
Dwgt = pdist2(X,Y, @(Xi,Xj) weuc(Xi,Xj,Wgts));


X = dotProdMat(:,1:100)'; 

k2 = 2; E = kron(eye(k2), ones(kOrbitSize)); kE_term = (k2*E - 1) - 0.5*(k2-1)*eye(kOrbitSize*k2);
kd = 2*kOrbitSize; % number of comparison elements
c = kd*(kd+1)/2; % scaling constant = number of lower triangular elements

distFun = @(XI,XJ)(regWdist1(XI, XJ, s, kE_term, c)); 
D = pdist2(X(1,:), X, distFun);

distFun = @(XI,XJ)(regWdist1(XI, XJ, s, kE_term, c)); 
D = pdist2(X(1:100,:), X(1:10,:), distFun);