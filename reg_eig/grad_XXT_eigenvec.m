% Optimized computation for grad to find different eigenvectors XX^T with eigenvalue lambda 

function dE = grad_XXT_eigenvec(W, XXt, d, k, s)

J = ones(k);
Id = eye(d);
WtW_J = W'*W - J;

dE = 2*((XXt - Id)^2)*W - (8/s^2)*exp(-(trace(WtW_J^2))/s^2)*sqrt(trace(WtW_J^2))*W*(WtW_J);

