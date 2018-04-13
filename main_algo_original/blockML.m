% Minimize wrt. M 
%
% Constraints:
% - double-stochasticity
% - orthogonality
% - sparsity
%
%
% tau: (gradient) descend step length
% beta: step length decrease constant
% lambda_vec =[lambda_1, lambda_2, lambda_3, lambda_4]. lambda_vec(4) is the 
% sparsity refularization constant (lambda4 in our formulation)

function M = blockML(MAX_ITER, tau, beta, M, D, w, Jk, vk, lambda_vec)

DtD = D'*D; % fixed, compute once
k = length(w);
Ik = eye(k);

for indM = 1:k % loop over each matrix k
    
    tauMi = tau; % re-initialize lambda for each matrix
    
    for iterNo = 1:MAX_ITER % loop over internal iterations
        Mi = M(:, :, indM);
        % Mt = permute(M, [2 1 3]);
        % MplusMt = Mi + Mi';
        
        grad_Mi = ...
            lambda_vec(1)*(sum(Dw1(w, M, k), 3)-DtD)*w(indM) + ...
            lambda_vec(2)*(sum(Mi, 2) - 1)*vk' + ...
            lambda_vec(2)*vk*(sum(Mi, 1) - 1) + ...
            lambda_vec(3)*(sum(M, 3) - Jk) + ...
            2*lambda_vec(4)*(Mi*Mi' - Ik)*Mi; % orthogonality constraint
        
        % lambda_vec(2)*(sum(MplusMt, 2) - 1)*vk' + ...
        
        
        %          grad_Mi = lambda_vec(1)*(sum(Dw1(w, M, k), 3)-DtD)*w(indM) + ...
        %             lambda_vec(2)*(squeeze(sum(sum(MplusMt, 2) - 1))*vk') + ...
        %             lambda_vec(3)*(sum(M, 3) - Jk);
        
        % grad_Mi = 2*lambda(1)*(sum(Dw1(w, M, k), 3) - DtD)*w(indM) + ...
        %    2*lambda(2)*(((sum(M, 3)+sum(TrML(M, k),3))*v-v)*v') + 2*lambda(3)*(sum(M, 3)-J);
        
        % M(:, :, indM) = project_pos(Mi - 2*lambdaMi*grad_Mi); % no need for sparsity
        M(:, :, indM) = project_pos(sthresh(Mi - tauMi*grad_Mi, 's', tauMi*lambda_vec(4)));
        
        % project in set of symmetric matrices M 
        M(:, :, indM) = project_sym(M(:, :, indM)); 
        
        tauMi = beta*tauMi;
    end
    
    
end