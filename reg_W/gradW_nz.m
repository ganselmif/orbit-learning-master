% Low, off-diagonal regularizer gradient

function dW = gradW_nz(W, k, s, Ik, R)

if nargin<4
    R = sparse(eye(k^2) + genT_opt(k));
    Ik = sparse(eye(k)); %Ik = eye(k);
end

vecG = vec(W'*W + eye(size(W, 2))*realmax);

I = -(2/s).*(vecG.*exp(-((vecG).^2)./s));

c = k^2; % scaling constant = number of lower triangular elements
dW = kron(Ik, W)*(R'*I)/c;