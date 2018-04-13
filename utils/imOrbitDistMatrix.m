% Visualize parts of distance matrix using binary indicator matrices. 
% 
% Use: inter-intra- orbit matrix distMat assumed to be a (k*nComp x k*nComp) 
% matrix of distances (i.e. nComp orbits of size k). ind_in and ind_out are 
% binary matrices indicating the locations of distMat to display.
% 
% TO-DO: randomize the orbit components to show. Now shows first k
% GE, CBMM/LCSL/MIT, gevang@mit.edu

function [h1, h2] = imOrbitDistMatrix(distMat, ind_in, ind_out, k, nComp)

if nargin<5, nComp = size(distMat, 1)/k; end % number of components to visualize

% k = 6; % orbit size
ind_both = zeros(size(ind_in));
ind_both(ind_in==1) = 1;
ind_both(ind_out==1) = 2;

% distMat(ind_both==0) = nan;
% ind_both(ind_both==0) = nan;

if nargout<=1
    h1 = figure;
    subplot 121; imagesc(ind_both(1:k*nComp, 1:k*nComp)); axis square;
    % subplot 132; imagesc(ind_out(1:k*nComp, 1:k*nComp)); axis square;
    subplot 122; imagesc(distMat(1:k*nComp, 1:k*nComp)); axis square;
elseif nargout==2
    h1 = figure; imagesc(distMat(1:k*nComp, 1:k*nComp)); axis square;
    h2 = figure; imagesc(ind_both(1:k*nComp, 1:k*nComp)); axis square;
end