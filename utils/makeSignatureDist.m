% Matrix and in- out- of orbit distances for invariant signature computation 
 

function [dist_in, dist_out, distMat, poolMat] = makeSignatureDist(O, D, poolMethod, distMethod, index_orbit, ind_in, ind_out)

%% Distance computation
% poolMethodNames = {'max', 'rms', 'mean', 'hist', 'moments', 'centmom', 'sumstats', 'cdf', 'histc'};
% distMethodNames = {'euclidean', 'cosine', 'seuclidean', 'chebychev', 'minkowski'};

% Pool and distance method
if nargin<4 || isempty(poolMethod), poolMethod = 'mearelu'; end 
if nargin<3 || isempty(distMethod), distMethod = 'euclidean'; end 

%% Normalized dot-product with dictionary
dotProdMat = normdotprod(O, D);

%% Pooling sover dictionaries (assuming 1 orbit in each)
[poolMat, ~] = poolOverTemplates2(dotProdMat, index_orbit, poolMethod, []); %, nBins, [minD, maxD]);
% poolMat = reshape(poolMat, [nOrbitSamples, nTrials])'; % each row belongs to the same test orbit
distMat = pdist2(poolMat, poolMat, distMethod);

%% In and Out of class/orbit distances
dist_in = distMat(ind_in);
dist_out = distMat(ind_out);
