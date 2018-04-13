% Update M matrices (with M sparsity)
%
%
% gamma: the refularization constant (lambda4 in our formulation)
% lambda: the proximal gradient constant
% beta: the proximal gradient annealing constant


function M = blockML_simple(MAX_ITER, lambda, beta, M, lambda_vec, gamma)

nMatrices = size(M,3);

k = size(M, 1);
Ik = eye(k);
vk = ones(k, 1); % vector of all ones of dim k

for indM = 1:nMatrices % loop over each matrix k
    
    lambdaMi = lambda; % re-initialize lambda for each matrix
    
    for iterNo = 1:MAX_ITER % loop over internal iterations
        
        Mi = M(:, :, indM);
        % Mt = permute(M, [2 1 3]);
        % MplusMt = Mi + Mi';
        
        M_constr_ortho = 0; % 2*(Mi*Mi' - Ik)*Mi;  % 4*(norm(Mi, 'fro').^2 - d)*Mi;
        
        %  %2*(sum(M, 3) - Jk);
        
        grad_Mi = lambda_vec(1)*(sum(Mi, 2) - 1)*vk' + ...
            lambda_vec(1)*vk*(sum(Mi, 1) - 1) + ...
            lambda_vec(2)*M_constr_ortho;
        
        %% Caley transform/Orthogonality Constraint
        % Wen, Yin, "A feasible method for optimization with orthogonality constraints"
        
        A = grad_Mi*Mi' - Mi*grad_Mi';
        Y = (Ik + A*iterNo/2)\(Ik-A*iterNo/2);
        
        % M(:, :, indM) = Y; 
        M(:, :, indM) = project_pos(Y);
        % M(:, :, indM) = project_pos(sthresh(Y, 's', lambdaMi*gamma));
        
        
        % M(:, :, indM) = Mi - lambdaMi*grad_Mi;
        % M(:, :, indM) = project_pos(Mi - 2*lambdaMi*grad_Mi);
        % M(:, :, indM) = project_pos(sthresh(Mi - lambdaMi*grad_Mi, 's', lambdaMi*gamma));
        
        
        %% orthogonality
        % M = bsxfun(@rdivide, M, sqrt(sum(M.*M,2)));
        
        %% double-stochasticity
        % M = (1/d)*Jk + M -(1/d)*Jk*M - (1/d)*M*Jk +(1/d*d)*(vk'*M*vk)*Jk;
        % M = M./sum(M(:,1));
        
        % M = blockML_SinkHorn(10, M);
        
        lambdaMi = beta*lambdaMi;
        
        dM = norm(Mi - M(:, :, indM), 'fro');
        fprintf('i: %d, dM: %f\n', iterNo, dM);
        
        
    end
    
    
end





