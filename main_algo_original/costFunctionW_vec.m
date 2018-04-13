% Vectorized cost function wrt. to w

function [J, gradW] = costFunctionW_vec(w, M, D, lambda)

% cached computations: compute once
DtD = D'*D;
k = length(w);
Mt = permute(M, [2 1 3]);
MplusMt = M + Mt; % M + M'

%% Cost function (differentiable terms depending on W) 
J = 0.5*lambda*sum(sum((sum(Dw1(w, M, k),3) - DtD).^2));
% J = norm(Dw1(w, M, k),3 - DtD, 'fro');

%% Gradient (unrolled)
for indM = 1:k % loop over vector dimension
    gradW(indM, 1) = lambda*(trace(M(:,:,indM)*sum(Dw1(w, Mt, k),3)) - 0.5*trace(DtD*MplusMt(:,:,indM)));
end