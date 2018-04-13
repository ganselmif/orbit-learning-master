% Commutator function and gradient

function [J, grad_W] = wComm(W, XXt, k)

WWt = W*W'/k; % divide by number of elements

J = - trace(comm(XXt, WWt)^2);
% grad_W = 4*comm(comm(WWt, XXt), XXt)*W;
grad_W = vec(4*comm2(WWt, XXt)*W/k); % maybe a bit faster