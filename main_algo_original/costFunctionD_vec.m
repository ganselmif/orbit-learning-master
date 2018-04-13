% Cost function and gradient with repsect to D
% 
% vecD is a vectorized D
% J is a float
% gradD is a vectorized gradient

function [J, gradD] = costFunctionD_vec(vecD, X, M, w, A, lambda1)

k = length(w);
d = length(vecD)/k;
D = reshape(vecD, d, k); % unroll variables for computations

%% project in unit ball D
% D  = project_unit(D);

%% Cached computations (move outside this wrapper!)
n = size(X, 2);
AAt = A*A';
XAt = X*A';
DtD = D'*D;

MplusMt = M + permute(M, [2 1 3]); % M + M'
sumwMMt = -sum(Dw1(w, MplusMt, k), 3);
sumwM = sum(Dw1(w, M, k), 3);

%% Cost function (differentiable terms depending on D)
J = (0.5/n)*(sum(sum((X - D*A).^2)) + lambda1*sum(sum((sumwM - DtD).^2)));

%% Gradient
gradD = (D*AAt-XAt)/n + lambda1*D*(sumwMMt + 2*(DtD));

% Re-roll gradient
gradD = gradD(:);