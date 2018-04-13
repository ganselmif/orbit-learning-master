% Check analytic vs. numerical gradient for (multiple) orbit regularizer
%
% % Example (how to test):
% 
% k = 20; Wo = rand(k, k); % need to use random matrix! 
% tic; testgradW(Wo, s, 1); toc 

function diff = testgradW_mult_x(W, setm)
e = 10^-5;

[d, K] = size(W);
k = K/length(unique(setm)); % number of groups/orbits

Jk = ones(k);
R = -Jk + (k)*eye(k);

% tic;
delta = zeros([d, K]);
for i=1:d
    for j=1:K
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (sregW_mult_cross(Wp, setm, k, Jk, R) - sregW_mult_cross(Wm, setm, k, Jk, R))/(2*e);
    end
end
numgrad = vec(delta);
% time_n = toc;

grad_W = sgradW_mult_cross(W, setm, k);
grad_W = grad_W(:); % vectorize

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
