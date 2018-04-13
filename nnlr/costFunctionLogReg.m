function [J, grad] = costFunctionLogReg(W_vec, X, yk, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, yk, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = size(X, 1);                     % number of points/inputs
% num_labels = size(yk, 2);         % number of output labels
layer_sizes = [size(X, 2) size(yk, 2)];
num_layers =  length(layer_sizes)-1; % number of layers (excluding output)

% Reshape W_vec back into the parameters W (cell array!)
W = paramVec2Mats(W_vec, layer_sizes, num_layers);
W = W{:};

% add bias and get sigmoid response
[h, ~] = nnFpropNode([ones(m,1) X], W); 
% h = sigmoid(X*W); 
% j = (2:size(W)); % j>=1

%% 1. Cost Function: k-dim (for each input m)
Jk = -sum(yk.*log(h) + (1-yk).*log(1-h), 2);
% Jk = -sum(yk.*log(h), 2); % Softmax regression

%% Regularization term 
% Jr = 0;
% for l = 1:num_layers-1
%     Jr = Jr + sum(sum(W{l}(:,2:end).^2));
% end
Jr = sum(sum(W(:, 2:end).^2)); % Frobenious squared  

%% Total cost: Cost averaged over m samples + regularization term
J = (1/m)*sum(Jk) + (lambda/(2*m))*Jr; 

%% 2. Gradient
grad = [];
W_grad = (1/m)*[ones(m,1) X]'*(h - yk); 
W_grad = W_grad';

% w. L2 Regularization (on the weights not the bias terms)
W_grad(:, 2:end) = W_grad(:, 2:end) + (lambda/m)*W(:, 2:end);

% Unroll gradients
grad = [grad; W_grad(:)];


