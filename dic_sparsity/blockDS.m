% Dictionary update via gradient descend
%
% tau: gradient learning rate
% lambda: reg(W) parameter

function D = blockDS(MAX_ITER, tau, beta, X, D, A, k, s, lambda)

% cached computations: compute once
[d, n] = size(X);
AAt = A*A';
XAt = X*A';

for iterNo= 1:MAX_ITER
    
    grad_D = (D*AAt-XAt)/n + lambda*reshape(gradW_opt_1(D, k, s), [d, k]);
    
    D = D - tau*grad_D;
    
    % projected gradient: project in unit ball the result of each iteration
    D  = project_unit(project_pos(D));
    
    tau = beta*tau;
    % D = project_unit_norm(D);
end