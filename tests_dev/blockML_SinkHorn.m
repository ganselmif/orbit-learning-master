% Update M matrices (with M sparsity)
%
%
% gamma: the refularization constant (lambda4 in our formulation)
% lambda: the proximal gradient constant
% beta: the proximal gradient annealing constant


function M = blockML_SinkHorn(MAX_ITER, M) %, gamma)
        
for iterNo = 1:MAX_ITER % loop over internal iterations
    Mo = M;
    M = bsxfun(@rdivide, M, sum(M,2));
    % fprintf('Iter: %d,\n', iterNo)
    M = bsxfun(@rdivide, M, sum(M,1));
    % disp(norm(M-Mo, 'fro'));
end
    
    




