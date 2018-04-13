function diff = testgradWCommColumns(W, X)

e = 10^-9;
[d, k] = size(W);
delta = zeros([d, k]);
Id = eye(k);

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
        delta(i,j) = 0;
        % loop over columns
        for q = 1:k
            R = Id(:,q)*(Id(:,q))';
            delta(i,j) = delta(i,j) + (-trace(comm(XXt, Wp*R*Wp')^2) + trace(comm(XXt, Wm*R*Wm')^2))/(2*e);
        end
    end
end

numgrad = vec(delta);

% loop over columns
grad_W = 0;
for q = 1:k
    R = Id(:,q)*(Id(:,q))';
    grad_W = grad_W + vec(4*comm(comm(W*R*W', XXt), XXt)*W*R);
end

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
