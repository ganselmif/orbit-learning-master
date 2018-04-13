% Regularizer for imposing same eigenvalues

function regEin = regE(W, XXt, d, k, s)

J = ones(k);
Id = eye(d);

regEin = trace(((XXt - Id)^2)*(W*W')) + exp(-(trace((W'*W - J)^2))/s^2);