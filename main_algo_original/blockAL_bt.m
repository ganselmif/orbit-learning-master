% Coding coefficients update using ISTA (proximal gradient)
%
% lambda: the regularization constant (lambda5 in our formulation)
% tau: (gradient) descend step length
% beta: line search parameter

% Includes backtracking line search (inexact)

function A = blockAL_bt(MAX_ITER, tau, beta, X, D, A, lambda)
% if isempty(tau),
%     a = 0.5; % default line backtracking constant
% else
%     a = tau; % use provided step
% end
% tau = 1; % default step length (for each iteration)

% cached computations: compute once
n = size(X, 2);
DtX = (D'*X);
DtD = (D'*D);

costFunc = @(p) (0.5/n)*(sum(sum((X - D*p).^2)));
objFunc = @(p) costFunc(p) + lambda*norm(p(:), 1);  % objective function (with sparsity norm)
objFuncVal(1) = objFunc(A);
e = 10^-4; % absolute tolerance

for iterNo = 1:MAX_ITER
    % nEval = 0;
    
    f = costFunc(A);   % Cost at current x
    
    %% Back-tracking line search
    while 1 && tau>10^-5
        
        %% Proximal Gradient
        grad_A = (DtD*A - DtX)/n;
        An = sthresh(A - tau*grad_A, 's', tau*lambda);
        
        fn = costFunc(An); % Cost at candidate x        
        if fn <= f + grad_A(:)'*(An(:) - A(:)) + (1/(2*tau))*sum(sum(An - A).^2)
            break;
        end
        tau = beta*tau;
        % nEval = nEval + 1; % number of function evalutions        
    end
    A = An;
    % disp([tau, nEval])
    
    %% termination condition (internal)
    objFuncVal(iterNo+1) = objFunc(A);
    if abs(objFuncVal(iterNo+1) - objFuncVal(iterNo))<e
        break
    end
    
end
 
% Optimal, dense solution: A = D\X;