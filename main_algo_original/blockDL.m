% Dictionary update through projected gradient descend
 
function D = blockDL(MAX_ITER, tau, beta, X, M, D, w, A, lambda1)

% cached computations: compute once
n = size(X, 2);
AAt = A*A';
XAt = X*A';

k = length(w);
MplusMt = M + permute(M, [2 1 3]); % M + M'
sumwMMt = -sum(Dw1(w, MplusMt, k), 3);

for iterNo= 1:MAX_ITER
    grad_D = (D*AAt-XAt)/n + lambda1*D*(sumwMMt + 2*(D'*D));

    % Note: (variant): constraint for sum to one in rows of Gramian
    % grad_D = (D*AAt-XAt)/n + 2*lambda6*D*(D'*D*vk - vk)*vk'; 
    
    D = D - tau*grad_D;
    
    % projected gradient: project in unit ball the result of each iteration
    D  = project_unit(project_pos(D));
    
    tau = beta*tau;
end