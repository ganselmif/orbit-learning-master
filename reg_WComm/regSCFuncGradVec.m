% Self-coherence term 
%
% See also: regWCommFuncGradVec.m regWCommColumnsFuncGradVec.m regWCommDetCovFuncGradVec.m 

function [J, gradW] = regSCFuncGradVec(vecW, k ,d)

%% Unroll variables and covariance
W = reshape(vecW, d, k);
% WWt = W*W'/k; % divide by number of elements
WtW = W'*W;

J = trace(WtW*WtW) - 2*trace(WtW) + d;
% J = trace(W'*(W*W')*W) - 2*trace(WtW)
gradW = 4*vec((W*WtW - W));