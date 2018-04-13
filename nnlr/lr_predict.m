function p = lr_predict(W, X)
%LR_PREDICT Prediction using logistic regression

%   p = LR_PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); 
% th = 0.5;
% p(sigmoid(X*theta) >= th) = 1; % predict 1)

%% Output units: add bias and get sigmoid response
% h = sigmoid(X*W');
[h, ~] = nnFpropNode([ones(m,1) X], W); 

%% Prediction based on max estimate
[~, p] = max(h, [], 2); % label class as the one with maximum prob