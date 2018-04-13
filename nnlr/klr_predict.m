function p = klr_predict(W, X)
%KLR_PREDICT Prediction using kernel logistic regression

%% Output units: add bias and get sigmoid response
% h = sigmoid(X*W');
[h, ~] = nnFpropNode(X, W'); 

%% Prediction based on max estimate
[~, p] = max(h, [], 2); % label class as the one with maximum prob