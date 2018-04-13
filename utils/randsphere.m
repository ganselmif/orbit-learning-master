% Random Points in an n-Dimensional Hypersphere
% 
% Returns an m x n array, X, in which each of the m rows has the n Cartesian 
% coordinates of a random point uniformly-distributed over the interior of 
% an n-dimensional hypersphere with radius r and center at the origin. 
% 
% Function 'randn' is initially used to generate m sets of n random 
% variables with independent multivariate normal distribution, with mean 
% 0 and variance 1. Then the incomplete gamma function, 'gammainc', 
% is used to map these points radially to fit in the hypersphere of finite 
% radius r with a uniform spatial distribution.
%
% Author: Roger Stafford - 12/23/05
% 
% Example:
% 
% N = 1000;
% % Unit ball
% X0 = randsphere(N, 2, 1); figure; scatter(X0(:,1), X0(:,2),'.'); axis equal;
% X0 = randsphere(N, 3, 1); figure; scatter3(X0(:,1), X0(:,2),X0(:,3),'.'); axis equal;
% % Unit Simplex
% X0 = rand(N, 2); figure; scatter(X0(:,1), X0(:,2),'r.'); axis equal;
% X0 = rand(N, 3); figure; scatter3(X0(:,1), X0(:,2),X0(:,3),'r.'); axis equal;

function X = randsphere(m, n, r)
 
X = randn(m, n);
s2 = sum(X.^2, 2);
X = X.*repmat(r*(gammainc(s2/2, n/2).^(1/n))./sqrt(s2), 1, n);
 