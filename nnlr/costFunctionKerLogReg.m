function [J, grad] = costFunctionKerLogReg(W_vec, X, yk, lambda)
% Kernel Logistic Regression
m = size(X, 1);  % number of instances 
layer_sizes = [size(X, 2) size(yk, 2)];
W = reshape(W_vec, layer_sizes);

[h, ~] = nnFpropNode(X, W');

%% 1. Cost Function: k-dim (for each input m)
Jk = -sum(yk.*log(h) + (1-yk).*log(1-h), 2);

%% Regularization term
% Jr = sum(sum(W'*X*W));
Jr = sum(diag(W'*X*W));

%% Total cost: Cost averaged over m samples + regularization term
J = (1/m)*sum(Jk) + (lambda/(m))*Jr;

%% 2. Gradient
grad = [];
W_grad = (1/m)*X*(h - yk);

W_grad = W_grad + 2*(lambda)*X*W/m;

% Unroll gradients
grad = [grad; W_grad(:)];