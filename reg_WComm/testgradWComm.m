% Test analytic gradient of the commutator function
%
% See: script_min_regWComm.m

function diff = testgradWComm(W, X)

e = 10^-9;
[d, k] = size(W);
delta = zeros([d, k]);

if nargin==1
    % random X
    N = k;
    X = rand([d, N]);
end

XXt = X*X'; % cached computation
for i=1:d
    for j=1:k
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (-trace(comm(XXt, Wp*Wp')^2) + trace(comm(XXt, Wm*Wm')^2))/(2*e);
    end
end

numgrad = vec(delta);
grad_W = vec(4*comm(comm(W*W', XXt), XXt)*W);
% grad_W = vec(4*comm2(W*W', XXt)*W); % maybe a bit faster

%vec(4*(X*X'*W*W'-W*W'*X*X')*W-8*X*X'*W*W'*X*X'*W);
% diff = delta-grad_W

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
