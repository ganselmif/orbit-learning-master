% Sigmoid function and derivative

% Example
% a = 10; x = (-10:0.1:10);
% [y, dydx] = sigm(x, a)
% figure; hold all;
% plot(x,y); plot(x, dydx);

function [y, dydx] = sigm(x, a)
if nargin<2, a = 1; end

y = 1 ./ (1 + exp(-a*x));
dydx = (y .* (1 - y));
