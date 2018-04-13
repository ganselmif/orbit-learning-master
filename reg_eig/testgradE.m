function diff = testgradE(W, X, d, k, s)

if nargin<5, s = 0.1; end 
if nargin<4, k = 6; end % ?
if nargin<3, d = 10; end % ?
if nargin<2, N = 10; X = rand([d, N]); end % ?
if nargin == 0, W = rand([d, k]); end

e = 10^-9;
delta = zeros([d, k]);

for i=1:d
    for j=1:k
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (regE(Wp, X, d, k, s) - regE(Wm, X, d, k, s))/(2*e);        
    end
end

numgrad = vec(delta);
grad_W = vec(grad_XXT_eigenvec(W, X, d, k, s));
% diff=delta-grad_W

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);