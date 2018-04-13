% Coding coefficients update using ISTA (proximal gradient)
%
% lambda: the regularization constant (lambda5 in our formulation)
% tau: (gradient) descend step length
% beta: step length decrease constant

function [A, objFuncValue] = blockAL(MAX_ITER, tau, beta, X, D, A, lambda)

% if nargin<7,
%     lambda = 0.1*norm(D*A, 'inf');
% end

%% Cached computations
n = size(X, 2);
DtX = (D'*X);
DtD = (D'*D);

% for convergence checks
e = 10^-4; % absolute tolerance
objFunc = @(p) (0.5/n)*(sum(sum((X - D*p).^2))) + lambda*norm(p(:), 1);
objFuncValue(1) = objFunc(A);

%% TO-CHECK: Constant update through gradient Lipschitz constant
% if isempty(tau), tau = sqrt(sum(sum(DtD.^2)))/n; beta = 1; end

for iterNo = 1:MAX_ITER
    
    grad_A = (DtD*A - DtX)/n;
    
    A = sthresh(A - tau*grad_A, 's', tau*lambda);
    tau = beta*tau;
    
    %% termination condition (for large internal changes)
    objFuncValue(iterNo + 1) = objFunc(A);    
    if abs(objFuncValue(iterNo+1) - objFuncValue(iterNo))<e
        % disp(iterNo)
        break
    end
    
end