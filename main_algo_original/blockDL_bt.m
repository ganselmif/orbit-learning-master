% Dictionary update through projected gradient descend
%
% D = blockDL_bt(MAX_ITER, tau, beta, X, M, D, w, A, lambda1)
%
% Includes backtracking line search (inexact)

% From blockDL.m

function D = blockDL_bt(MAX_ITER, tau, beta, X, M, D, w, A, lambda1)

if isempty(tau), 
    a = 0.5; % default line backtracking constant
else
    a = tau; % use provided step
end
tau = 1; % default step length (for each iteration)

%% Cached computations
% n = size(X, 2); k =length(w);
[k, n] = size(A);
AAt = A*A';
XAt = X*A';
MplusMt = M + permute(M, [2 1 3]); % M + M'
sumwMMt = -sum(Dw1(w, MplusMt, k), 3);

%% Cost function (differentiable terms depending on D)
sumwM = sum(Dw1(w, M, k), 3);
costFunc = @(p) (0.5/n)*(sum(sum((X - p*A).^2)) + lambda1*sum(sum((sumwM - p'*p).^2)));
objFuncVal(1) = costFunc(D);
e = 10^-4; % absolute tolerance

for iterNo = 1:MAX_ITER
    tau_i = tau; % initial step length for back-tracking
    nEval = 0; % number of function evalutions
    
    %% Gradient, descend and unit ball projection
    grad_D = (D*AAt-XAt)/n + lambda1*D*(sumwMMt + 2*(D'*D));
    % Note: (variant): constraint for sum to one in rows of Gramian
    % grad_D = (D*AAt-XAt)/n + 2*lambda6*D*(D'*D*vk - vk)*vk';
    
    f = costFunc(D);   % Cost at current x
    
    %% Back-tracking line search
    while 1 && tau_i>10^-5
        nEval = nEval + 1;
        
        Dn =  project_unit(D - tau_i*grad_D); % projected (in unit ball) gradient
        % Dn =  D - tau_i*grad_D;
        
        fn = costFunc(Dn); % Cost at candidate x
        
        if fn <= f - a*tau_i*sum(sum(grad_D.^2))
            break
        end
        tau_i = beta*tau_i;
        
    end
    D = Dn;
    % fprintf('%d %d %f\n', iterNo, nEval, tau_i);
    
    %% termination condition (internal)
    objFuncVal(iterNo+1) = costFunc(D);
    if abs(objFuncVal(iterNo+1) - objFuncVal(iterNo))<e
        break
    end
    
end

% function S = sum_square(X)
% S = sum(sum(X.^2));

