% Unsupervised learning using toy data (MAIN SCRIPT)
%
% Dependencies: blokAL, blockDL, Dw1, project_pos, projet_unit, sthresh
%
% Data generation: datasets_groups directory
%
% MATLAB dependencies: fminunc (optional)
%
% Source: main_algo.m main_algoE.m

clear; % clear all;
close all hidden;

%% Data X (d x n)
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random

N = 100; % number of (untransformed) data

filename_data = 'CyclicGroup6.txt';
X = genGroupData(filename_data, N);
P = importPermutationsToMat(filename_data);
[d, n] = size(X);
k = n/N; % orbit size

% % % unit-norm data
% normx = sqrt(sum(X.^2, 1));
% X = bsxfun(@times, X, 1./normx);

% Pm(:,:,1,:) = P; figure; montage(Pm);

%% Random inizialization of all variables
% rng('shuffle'); % re-initialize random seed generator (for random initializations)!

% Dictionary (d x k)
D = randArrayInRange([d, k, 1], 0, 1);
D = project_unit(D); % project in unit ball

% Coding coefficients (k x n)
A = randn(k, n); % A = D\X;

%% Gradient desend learning rate (tau)

% learning rates
beta = 0.5; % line search/decrease in step size  ([0.1, 0.8])
betaA = beta; tauA = 0.1;
betaD = beta; tauD = 0.1;


%% Regularization constants (lambda)
c = n;                     % controls the order of magnitude of the sparsity terms
% (e.g. based on number of data points)
lambdaD = 0;              % reg(W) term on D 
lambdaA = 1; %1/(c*sqrt(d));  % A-sparsity (from Mairal paper OR gamma: from Boyd/Lasso code


%% Main ALS method (calling suboptimizations in M, w, D, A
ITER_MAX_INNER = 20;  % num of iterations inside each optimization variable block
ITER_MAX_ALS = 200;   % passes in ALS iterations
e = 10^-8;            % tolerance/sensitivity constant for ALS convergence

% A = sparse(A); % can work with sparse matrices for efficiency
normfro = @(p) sqrt(sum(sum(p.^2))); % Frobenius norm, fast 

%%  Objective function
s = 0.001; % reg(W) value
f(1) = orbLearnObjectiveFunctionS(X, D, A, lambdaA, lambdaD, k, s);


for iterNo = 1 : ITER_MAX_ALS
    
    Ao = A; Do = D; 
    
    D = blockDS(ITER_MAX_INNER, tauD, betaD, X, D, A, k, s, lambdaD);
    % D = blockDL_f(ITER_MAX_INNER, X, M, D, w, A, lambda_vec(1), 'fminunc');  % Optimized by MATLAB's fminunc
    
    A = blockAL(ITER_MAX_INNER, tauA, betaA, X, D, A, lambdaA);
    % A = blockAL_bt(ITER_MAX_INNER, tauA, betaA, X, D, A, lambda_vec(5)); % Line back-tracking
    
    %% convergence track
    % objective function value
    f(iterNo+1) = orbLearnObjectiveFunctionS(X, D, A, lambdaA, lambdaD, k, s);
    dA(iterNo)= normfro(Ao-A); 
    dD(iterNo)= normfro(Do-D);
    
    fprintf('i: %d, F: %f, A: %f, D: %f\n', iterNo, f(iterNo), dA(iterNo), dD(iterNo))
    
    % Convergence check
    if abs(f(end)-f(end-1)) < e
        break
    end    
    
end

%% Plots and Figures
if 1
    % convergence plot
    ITER_MAX_ALS = length(dA);
    figure;
    subplot 131; semilogx(1:ITER_MAX_ALS, f(2:end), '.-'); axis tight; title('F');
    subplot 132; semilogx(1:ITER_MAX_ALS, dD,'.-'); axis tight; title('D')
    subplot 133; semilogx(1:ITER_MAX_ALS, dA, '.-'); axis tight; title('A')     
           
    figure;
    % dictionary
    subplot(3,2,1); imagesc(D); colorbar; title('Dictionary')
    
    % data
    subplot(3,2,2); imagesc(X); colorbar; title('Data')
    
    %gramian
    subplot(3,2,3); imagesc(D'*D); colorbar; title('Gramian')
    
    %A
    subplot(3,2,4); imagesc(A); colorbar; title('Coefficients')
    
    
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
