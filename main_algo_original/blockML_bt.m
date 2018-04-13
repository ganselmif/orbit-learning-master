% Minimize wrt. M
%
% Constraints:
% - double-stochasticity
% - orthogonality
% - sparsity
%
% tau: (gradient) descend step length
% beta: step length decrease constant
% lambda_vec =[lambda_1, lambda_2, lambda_3, lambda_4]. lambda_vec(4) is the
% sparsity refularization constant (lambda4 in our formulation)
%
% Includes backtracking line search (inexact)

% From blockML.m

function M = blockML_bt(MAX_ITER, tau, beta, M, D, w, lambda_vec)

DtD = D'*D; % fixed, compute once
k = length(w);
Ik = eye(k);
vk = ones(k, 1);
Jk = ones(k, k);

%% Cost function (differentiable terms depending on M)
costFunc = @(p) 0.5*(lambda_vec(1)*norm(sum(Dw1(w, p, k), 3) - DtD, 'fro')^2 + ...
    lambda_vec(2)*termPermuM(p) + ...
    lambda_vec(3)*norm(sum(p, 3) - Jk, 'fro')^2 + ...
    lambda_vec(4)*termOrthoM(p));
objFunc = @(p) costFunc(p) + lambda_vec(4)*norm(p(:), 1);  % cost function (with sparsity norm)

objFuncVal = repmat({objFunc(M)}, [1 k]);
e = 10^-4; % absolute tolerance

for indM = 1:k % loop over each matrix k
    
    tauMi = tau; % re-initialize lambda for each matrix
    
    for iterNo = 1:MAX_ITER % loop over internal iterations
        
        Mtemp = M; % temporary/current M
        f = costFunc(Mtemp);   % Cost at current x
        
        %% Back-tracking line search
        nEval = 1;
        while 1 && tauMi>10^-4
            
            Mi = Mtemp(:, :, indM);
            
            grad_Mi = ...
                lambda_vec(1)*(sum(Dw1(w, Mtemp, k), 3)-DtD)*w(indM) + ...
                lambda_vec(2)*(sum(Mi, 2) - 1)*vk' + ...
                lambda_vec(2)*vk*(sum(Mi, 1) - 1) + ...
                lambda_vec(3)*(sum(Mtemp, 3) - Jk) + ...
                2*lambda_vec(4)*(Mi*Mi' - Ik)*Mi; % orthogonality constraint
            
            % positivity and sparisty
            Mn = project_pos(sthresh(Mi - tauMi*grad_Mi, 's', tauMi*lambda_vec(4)));
            % project in set of symmetric matrices M
            Mn = project_sym(Mn);
            
            Mtemp(:,:,indM) = Mn;
            
            fn = costFunc(Mtemp); % Cost at candidate x
            if fn <= f + grad_Mi(:)'*(Mi(:) - Mn(:)) + (1/(2*tauMi))*sum(sum(Mn - Mi).^2)
                break;
            end
            
            tauMi = beta*tauMi;
            nEval = nEval + 1; % number of function evalutions
        end
        
        M(:, :, indM) = Mn;
        % fprintf('%d %d %f\n', iterNo, nEval, tauMi);
        
        %% termination condition (internal)
        objFuncVal{indM}(iterNo + 1) = objFunc(M);
        if abs(objFuncVal{indM}(iterNo + 1) - objFuncVal{indM}(iterNo))<e
            % fprintf('%d %d %f\n', indM, iterNo, tauMi);
            break
        end
    end
    
    
end