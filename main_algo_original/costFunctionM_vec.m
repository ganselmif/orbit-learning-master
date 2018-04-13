% Vectorized cost and gradient depending on M
% 
% vecM: a vectorized version of M  (rowise, slice wise
% lambda_vec: [lambda_1, lambda_2, lambda_3, lambda_4]

function [J, gradM] = costFunctionM_vec(vecM, D, w, lambda_vec)

[~, k] = size(D);
DtD = D'*D; % fixed, compute once
% k = length(w);
Ik = eye(k);
vk = ones(k, 1);
Jk = ones(k, k);

M = reshape(vecM, k, k, k); % unroll variables for computations

%% Cost function (differentiable terms depending on M)
J = 0.5*(lambda_vec(1)*norm(sum(Dw1(w, M, k), 3) - DtD, 'fro')^2 + ...
    lambda_vec(2)*termPermuM(M) + ...
    lambda_vec(3)*norm(sum(M, 3) - Jk, 'fro')^2 + ...
    lambda_vec(4)*termOrthoM(M));

%% Gradient wrt. M
for indM = 1:k % loop over each matrix k
    % lambdaMi = lambda; % re-initialize lambda for each matrix
    Mi = M(:, :, indM);
    
    gradM(:,:,indM) = lambda_vec(1)*(sum(Dw1(w, M, k), 3) - DtD)*w(indM) + ...
        lambda_vec(2)*(sum(Mi, 2) - 1)*vk' + ...
        lambda_vec(2)*vk*(sum(Mi, 1) - 1) + ...
        lambda_vec(3)*(sum(M, 3) - Jk) + ...
        2*lambda_vec(4)*(Mi*Mi' - Ik)*Mi; % orthogonality constraint
    
    % M(:, :, indM) = project_pos(sthresh(Mi - 2*lambdaMi*grad_Mi, 's', lambdaMi*gamma));
    % lambdaMi = beta*lambdaMi;
end

% Re-roll gradient
gradM = gradM(:);


