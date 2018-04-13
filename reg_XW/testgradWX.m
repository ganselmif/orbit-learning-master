% Test analytic gradient of the reg(WX) function
%
% See: script_min_regWX.m

function diff = testgradWX(W, X, d, k, s)
if nargin<5, s = 0.1; end 
if nargin<4, k = 6; end % ?
if nargin<3, d = 10; end % ?
if nargin<2, N = 10; X = rand([d, N]); end % ?
if nargin == 0, W = rand([d, k]); end

e = 10^-9;
delta = zeros([d, k]);
E = kron(eye(k), ones(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

for i=1:d
    for j=1:k
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (regW_fixed(X'*Wp, k, s, kE_term) - regW_fixed(X'*Wm, k, s, kE_term))/(2*e);
     end
end

numgrad = vec(delta);
grad_W = gradWX_opt_1_fixed(W, X, k, s);

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
