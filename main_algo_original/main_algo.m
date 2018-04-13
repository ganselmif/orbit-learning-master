% Unsupervised orbit learning using toy data (MAIN SCRIPT)
%
% Dependencies: blokAL, blockDL, blockML, blockwL, Dw1, project_pos,
%               projet_unit, sthresh, TrML, Mck, ...\
%
% Data generation: datasets_groups directory
%
% MATLAB dependencies: fminunc (optional)
%
% Sources:
% - ISTA/proximal gradient from https://web.stanford.edu/~boyd/papers/prox_algs/lasso.html
% -

clear; % clear all;
close all hidden;

k = 6;   % dictionary size
d = 6;   % data dimension and for now also data number...?

%% Data X (d x n)
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random

N = k; % number of (untransformed) data

% c = linspace(0.1, 1, d); X = gallery('circul', c);
% X = genRandCircData(d, N, 'random');

filename_group = 'CyclicGroup6.txt';
X = genGroupData(filename_group, N);
P = importPermutationsToMat(filename_group);
n = size(X, 2);

% % % unit-norm data
% normx = sqrt(sum(X.^2, 1));
% X = bsxfun(@times, X, 1./normx);

% Pm(:,:,1,:) = P; figure; montage(Pm);

%% Random inizialization of all variables
% rng('shuffle'); % re-initialize random seed generator (for random initializations)!

% D = arraygenL([d, k],1,0,1);
a = 0; b = 1;

% Dictionary (d x k)
D = randArrayInRange([d, k, 1], a, b);
D = project_unit(D); % project in unit ball

% w (k x 1)
w = randArrayInRange([k, 1, 1], a, b);
% M (k x k x k)
M = randArrayInRange([k, k, k], a, b);

% Coding coefficients (k x n)
A = randn(k, n); % A = D\X;

%% Gradient desend learning rate (tau)
beta = 0.5; % line search/decrease in step size  ([0.1, 0.8])
betaM = beta; tauM = 0.1; % descend step size (or line search constant) [0.1, 0.5))
betaA = beta; tauA = 0.1;
betaw = beta; tauW = 0.3;
betaD = beta; tauD = 0.3;

% Old constants
% betaM = beta; tauM = 0.1;   % descend step size (or line search constant) [0.1, 0.5))
% betaA = beta; tauA = 0.1;
% betaw = beta; tauW = 0.1;
% betaD = beta; tauD = 0.1;

%% Regularization constants (lambda)

c = n;                     % controls the order of magnitude of the sparsity terms
% (e.g. based on number of data points)
lambda1 = 1;               % Gramian/Latin square constraint
lambda2 = 0.5;             % Double stochasticity (sum to one)
lambda3 = 10*lambda2;      % All ones
lambda4 = 20/(c*sqrt(k));  % M-sparsity AND M-orthogonality regularization
lambda5 = 1/(c*sqrt(d));   % A-sparsity (from Mairal paper OR gamma: from Boyd/Lasso code

% Old parameters
% lambda1 = 1;
% lambda2 = 0.5;
% lambda3 = 5*lambda2;   % lambda2 = 3; lambda3 = 35;
% lambda4 = 2/sqrt(k);
% lambda5 = 1.2/sqrt(d);

%% Main ALS method (calling suboptimizations in M, w, D, A
ITER_MAX_INNER = 20;  % num of iterations inside each optimization variable block
ITER_MAX_ALS = 200;   % passes in ALS iterations
e = 10^-4;            % tolerance/sensitivity constant for ALS convergence

% constant matrices, vectors
Jk = ones(k);    % matrix of all ones of dim k X k
vk = ones(k, 1); % vector of all ones of dim k

% Objective function
lambda_vec = [lambda1; lambda2; lambda3; lambda4; lambda5];
f(1) = orbLearnObjectiveFunction(X, D, A, w, M, lambda_vec);

% A = sparse(A); % can work with sparse matrices for efficiency
normfro = @(p) sqrt(sum(sum(p.^2))); % Frobenius norm, fast 

for iterNo = 1 : ITER_MAX_ALS
    
    %% update each block alternatively
    Mo = M; Ao = A; Do = D; wo = w;
    M = blockML(ITER_MAX_INNER, tauM, betaM, M, D, w, Jk, vk, lambda_vec(1:4));
    % M = blockML_bt(ITER_MAX_INNER, tauM, betaM, M, D, w, lambda_vec(1:4)); % Line back-tracking
    
    w = blockwL(ITER_MAX_INNER, tauW, betaw, M, D, w, lambda_vec(1));
    % w = blockwL_f(ITER_MAX_INNER, M, D, w, lambda_vec(1), 'fminunc');    % Optimized by MATLAB's fminunc
    % w = blockwL_bt(ITER_MAX_INNER, tauW, betaw, M, D, w, lambda_vec(1)); % Line back-tracking
    
    D = blockDL(ITER_MAX_INNER, tauD, betaD, X, M, D, w, A, lambda_vec(1));
    % D = blockDL_f(ITER_MAX_INNER, X, M, D, w, A, lambda_vec(1), 'fminunc');    % Optimized by MATLAB's fminunc
    % D = blockDL_bt(ITER_MAX_INNER, tauD, betaD, X, M, D, w, A, lambda_vec(1)); % Line back-tracking
    
    A = blockAL(ITER_MAX_INNER, tauA, betaA, X, D, A, lambda_vec(5));
    % A = blockAL_bt(ITER_MAX_INNER, tauA, betaA, X, D, A, lambda_vec(5)); % Line back-tracking
    
    %% convergence track
    % objective function value
    f(iterNo+1) = orbLearnObjectiveFunction(X, D, A, w, M, lambda_vec);
    dM(iterNo)= normfro(Mo(:)-M(:));
    dA(iterNo)= normfro(Ao-A); 
    dD(iterNo)= normfro(Do-D);
    dW(iterNo)= normfro(wo-w);
    
    fprintf('i: %d, F: %f, M: %f, A: %f, D: %f, w: %f\n', ...
        iterNo, f(iterNo), dM(iterNo), dA(iterNo), dD(iterNo), dW(iterNo))
    
    % Convergence check
    if abs(f(end)-f(end-1)) < e
        break
    end
    
    
end

%% Plots and Figures
if 1
    % convergence plot
    ITER_MAX_ALS = length(dM);
    figure;
    subplot 231; semilogx(1:ITER_MAX_ALS, f(2:end), '.-'); axis tight; title('F');
    subplot 233; semilogx(1:ITER_MAX_ALS, dM, '.-'); axis tight; title('M')
    subplot 234; semilogx(1:ITER_MAX_ALS, dD,'.-'); axis tight; title('D')
    subplot 235; semilogx(1:ITER_MAX_ALS, dW, '.-'); axis tight; title('w')
    subplot 236; semilogx(1:ITER_MAX_ALS, dA, '.-'); axis tight; title('A')
    
    % %plot dictionary
    % figure;
    % hold on
    % for j = 1:d
    %  plot(D(:,j))
    % end
    
    % figure;
    % nRow = nextpow2(k); nCol = ceil(k/nRow);
    % for indMat=1:k
    %     % subplot(nRow, nCol, i)
    %     subaxis(nRow, nCol, indMat, 'Spacing', 0.02, 'Padding', 0, 'Margin', 0);
    %     imagesc(M(:,:,indMat)); axis square; axis off;
    % end
    % % colorbar; title(sprintf('M%d', i));
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    % matrix-sum-to-one plots
    figure; % figure('name','M');
    nCol = nextpow2(k); nRow = ceil(k/nCol);
    for indMat=1:k
        subplot(nRow, nCol, indMat)
        Mi = [M(:, :, indMat); sum(M(:,:,indMat), 1)];
        Mi = [Mi, [sum(M(:,:,indMat), 2); nan]];
        % sum(M(:,:,i), 1)
        % sum(M(:,:,i), 2)
        imagesc(Mi); axis square; colorbar;
    end
    title(sprintf('M%d', indMat));
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    % orthogonality plots
    figure;
    nCol = nextpow2(k); nRow = ceil(k/nCol);
    for indMat=1:k
        try
            subaxis(nRow, nCol, indMat, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
        catch ME
            subplot(nRow, nCol, indMat)
        end
        imagesc(M(:,:,indMat)*M(:,:,indMat)'); axis square; colorbar;
    end
    title(sprintf('MMt,d', indMat));
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    
    % Latin Square with pseudo/fixed-values (for visualization only)
    Wk = 1:k; sumK = 0;
    for indMat = 1:k
        sumK = sumK + Wk(indMat)*M(:,:,indMat);
    end
    
    figure;
    % dictionary
    subplot(3,2,1); imagesc(D); colorbar; title('Dictionary')
    
    % vector w (should be one column of the gramian)
    subplot(3,2,2); imagesc(w); colorbar; title('w')
    
    % latin square constraint, should be the all ones matrix
    subplot(3,2,3); imagesc(sum(M,3)); colorbar; title('Latin square constraint (=J)')
    
    % data
    subplot(3,2,4); imagesc(X); colorbar; title('Data')
    
    subplot(3,2,5); imagesc(D'*D); colorbar; title('Gramian')
    
    % (fake) Latin Square
    subplot(3,2,6); imagesc(sumK); colorbar; title('LS'); % colormap jet
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
end

% pause;

%% Quantitative Comparisons
% clc;
disp(' ');
disp('----------');
fprintf('Reconstruction = %f\n', (1/n)*norm(X-D*A, 'fro').^2);
fprintf('(Random A) Reconstruction = %f\n', (1/n)*norm(X-D*rand(k, n), 'fro').^2);
fprintf('(Random D) Reconstruction = %f\n', (1/n)*norm(X-project_unit(rand(d, k))*A, 'fro').^2);
disp('----------');
m = mean(mean(sum(M,3))); % Mn = M./m; % normalized
fprintf('Latin square constraint = %f\n', norm(sum(M,3)./m-Jk, 'fro'));
fprintf('M constraint = %f\n', termPermuM(M./m));
fprintf('Orthogonality M = %f\n', termOrthoM(M));
disp('----------');
fprintf('Sparsity M, |M|_0 = %f\n', sum(M(:)==0)/numel(M));
fprintf('Sparsity A, |A|_0 = %f, |A|_1 = %f\n', sum(A(:)==0)/numel(A), sum(abs(A(:)))/numel(A));
disp('----------');
% Is resulting dictionary a group?
fprintf('Group structure, std(sum(Dt*D)) = %f, %f)\n', std(sum(D'*D, 1)), std(sum(D'*D, 2)));

%if exist(filename_group, 'var')
try
    %% Invariance Test/Comparison signatures
    nTrials = 100; % number of random templates to use
    [ds_in, ds_out, Wo] = ComparisonGD(filename_data, D, false, nTrials);
    nBins = 40;
    dispComparisonGD(ds_in, ds_out, nBins);
    
catch ME
    
end
%close all
