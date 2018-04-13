% Main script: Orbit learning using toy data

% Dependencies: blokAL, blockDL, blockML, blockwL, Dw1, project_pos, 
%               projet_unit, sthresh, TrML, Mck, ...
% Source: ISTA/proximal gradient from https://web.stanford.edu/~boyd/papers/prox_algs/lasso.html

clear; % clear all;
close all;

k = 5; 

% M (k x k x k)
a = 0; b = 1;
% M = randArrayInRange([k, k, k], a, b); 
M = randArrayInRange([k, k, 1], a, b); 


% constant matrices, vectors
Jk = ones(k);    % matrix of all ones of dim k X k
vk = ones(k, 1); % vector of all ones of dim k

% A = (1/k)*Jk + M -(1/k)*Jk*M - (1/k)*M*Jk +(1/k*k)*(vk'*M*vk)*Jk; 
% A = A./sum(A(:,1));


beta = 0.5; 
lambdaM = 0.1; betaM = beta;

%% Regularization constants
gamma = 0.1;

lambda0 = 1; % orthogonality 
lambda2 = 1; % stoch
lambda3 = 0; % latin 


%% Main ALS method (calling suboptimizations in M, w, D, A
MAX_ITER = 100;  % blocks max iterations
% R = 100;        % main algo max iteration

% Objective function
lambdas = [lambda2 lambda0];
% f = orbLearnObjectiveFunction(X, D, A, w, M, lambdas);

%for iterNo = 1 : R

%% update each block alternatively
%    Mo = M;
% Ms = blockML_SinkHorn(MAX_ITER, M); %, gamma)

% M = blockML_simple(MAX_ITER, lambdaM, betaM, M, Jk, lambdas(1:2));

% M = blockML_simple(MAX_ITER, lambdaM, betaM, M, Jk, gamma);

% M = blockML_simple(MAX_ITER, lambdaM, betaM, M, Jk, vk);
Mo = M;
M = blockML_simple(MAX_ITER, lambdaM, betaM, Mo, lambdas, gamma);
Mn = M./max(M(:)); 
% M1 = blockM1(3600, lambdaM, betaM, Mo, vk, lambda2, lambda0, gamma, k); 

figure;
subplot 121; imagesc(Mn(:,:,1)); axis square; colorbar;
subplot 122; imagesc(Mn(:,:,1)*Mn(:,:,1)'); axis square; colorbar;
% subplot 122; imagesc(M1(:,:,1)); axis square; colorbar;
% title(sprintf('M%d', i));


% figure;
% nCol = nextpow2(k); nRow = ceil(k/nCol);
% for i=1:k
%     subplot(nRow, nCol, i)
%     imagesc(M(:,:,i)); axis square; colorbar;
%     title(sprintf('M%d', i));
% end
% colorbar;


% figure;
% % dictionary
% subplot(3,2,1); imagesc(D); colorbar; title('Dictionary')
% 
% % vector w (should be one column of the gramian)
% subplot(3,2,2); imagesc(w); colorbar;subplot 121; imagesc( title('w')
% 
% % latin square constraint, should be the all ones matrix
% subplot(3,2,3); imagesc(sum(M,3)); colorbar; title('Latin square constraint (=J)')
% 
% % data
% subplot(3,2,4); imagesc(X); colorbar; title('Data')
% 
% subplot(3,2,5); imagesc(D'*D); colorbar; title('Gramian')


% %% Quantitative Comparisons
% m = mean(mean(sum(M,3))); % Mn = M./m; % normalized
% % fprintf('Reconstruction = %d\n', norm(X-D*A,'fro'));
% fprintf('Latin square constraint = %d\n', norm(sum(M,3)./m-Jk,'fro'));
% fprintf('M constraint = %d\n', Mck(M./m, vk, d));
% fprintf('Sparsity M = %d\n', sum(abs(M(:))));
% % fprintf('Sparsity A = %d\n', norm(A,1));

