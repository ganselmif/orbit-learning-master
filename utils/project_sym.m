function W = project_sym(W)
% PROJECT_SYM Project a matrix W onto the set of symmetric matrices, i.e.
% such that W - W^T = 0 

W = (W + W')/2;
