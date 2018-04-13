% Utility function
% 
% Creates the logical MN x MN matrices of inter- and intra- distances for 
% M orbits of N elements each. The matrices index the locations of unique
% distances (excluding self) in a MX x MN matrix. 
%
% See also: makeSignatureDist.m

% GE, CBMM/LCSL/MIT, gevang@mit.edu
 
function [ind_in, ind_out] = indMatInterIntraOrbit(nOrbits, nSamplesPerOrbit) 

%% Util matrices for in-orbit and Out-of-orbit distributions
% Block-diagonal ground truth matrix (for inner-outer distances)
M = kron(eye(nOrbits), ones(nSamplesPerOrbit));

% Symmetric matrices/keep only upper triangular for all pairwise distances
tril_ones_n = tril(ones(nSamplesPerOrbit*nOrbits))~=1;

ind_in = tril_ones_n.*M==1; % In-orbit distances
ind_out = tril_ones_n.*(1-M)==1; % Out-of-orbit distances