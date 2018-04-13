% Self-coherence (or frame potential) of a d x k dictionary matrix 

function [J, grad_W] = wSelfCoherence(W, k)

% W = project_unit_norm(W);

WtW = W'*W; % Gramian of dictionary
% WtW = normdotprod(W, W); % TO-DO: Normalized dot products 

J = trace(WtW*WtW) - 2*trace(WtW) + k; 
% J = sum(sum((WtW - eye(k)).^2)); 

grad_W = 4*(W*WtW - W); % gradient 