function W = project_unit_norm(W)
% PROJECT_UNIT Project a dictionary (matrix of column elemnts) to have 
% unit-norm elements per column (column normalization) 

% with eps reg. for zero-division 
norm_func = @(x)(bsxfun(@rdivide, x, max(sqrt(sum(x.^2, 1)), eps)));

% norm_func = @(x)normc(x); %% NN toolbox function 

W = norm_func(W);