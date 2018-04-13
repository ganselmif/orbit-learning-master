function Xu = project_pca(X, nComp)

nSamples = size(X, 1);
meanX = sum(X, 1)/nSamples; % empirical (sample) mean
Xcent = bsxfun(@minus, X, meanX); % deviations from the mean
covX = Xcent'*Xcent/(nSamples-1); % data covariance, normalized by (N-1)
[U, ~] = svd(covX);

%% Truncated eigenvectors
Utrunc = U(:,1:nComp);

%% Project data to U
Xu = Xcent*Utrunc;