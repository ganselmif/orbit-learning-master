% Update of the w vector
% Note: (with sparsity, may be not necessary)
%
% tau: (gradient) descend step length
% beta: line search parameter

% Includes backtracking line search (inexact)
% From blockWl.m

function w = blockwL_bt(MAX_ITER, tau, beta, M, D, w, lambda1)
if isempty(tau),
    a = 0.5; % default line backtracking constant
else
    a = tau; % use provided step
end
tau = 1; % default step length (for each iteration)

% cached computations: compute once
DtD = D'*D;
k = length(w);
Mt = permute(M, [2 1 3]);
MplusMt = M + Mt; % M + M'

%% Cost function (differentiable terms depending on w)
costFunc = @(p) 0.5*lambda1*sum(sum((sum(Dw1(p, M, k),3) - DtD).^2));
objFuncVal(1) = costFunc(w);
e = 10^-4; % absolute tolerance

for iterNo = 1:MAX_ITER
    
    tau_i = tau; % initial step length for back-tracking
    nEval = 0; % number of function evalutions
    
    for indM = 1:k % loop over vector dimension
        % grad_wi = 2*lambda1*(trace(M(:,:,indM)*sum(Dw1(w,Mt,k),3)) - trace(DtD*MplusMt(:,:,indM)));
        grad_w(indM, 1) = lambda1*(trace(M(:,:,indM)*sum(Dw1(w, Mt, k),3)) - 0.5*trace(DtD*MplusMt(:,:,indM)));
        
    end
    
    f = costFunc(w);   % Cost at current x
    
    %% Back-tracking line search
    while 1 && tau_i>10^-5
        nEval = nEval + 1;
        % wn = project_unit(w - tau_i*grad_w);
        wn = w - tau_i*grad_w;
        
        fn = costFunc(wn); % Cost at candidate x
        if fn <= f - a*tau_i*sum(sum(grad_w.^2))
            break;
        end
        tau_i = beta*tau_i;
        
    end
    w = wn;
    % fprintf('%d %d %f\n', iterNo, nEval, tau_i);
    
    %% termination condition (internal)
    objFuncVal(iterNo+1) = fn; % costFunc(w);
    if abs(objFuncVal(iterNo+1) - objFuncVal(iterNo))<e
        break
    end
    
    
end
% w  = project_unit(w); % w = w./norm(w);