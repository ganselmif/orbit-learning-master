% Test analytic gradient of regCN
%
% See: 

function diff = testgradWCN(W, maxp)

e = 10^-9;

if nargin==1, maxp = 5; end
if nargin==0; d = 100; k = 60; W = rand([d, k]); end

[d, k] = size(W); 
delta = zeros([d, k]);

for i=1:d
    for j=1:k
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (regCN(Wp,maxp,d,k)-regCN(Wm,maxp,d,k))/(2*e);
    end
end

numgrad = vec(delta);
grad_W = vec(gradWCN(W, maxp, d, k));

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
