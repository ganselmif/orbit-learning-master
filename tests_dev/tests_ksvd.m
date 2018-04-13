% Baseline tests

clear; % clear all;
close all hidden;

k = 100;   % dictionary size
d = 6;    % data dimension and for now also data number...?

%% Data X (d x n)
rng(0, 'twister'); % Fix seed for reproducible results

N = 10*k;
% c = linspace(0.1, 1, d); X = gallery('circul', c);

% X = genRandCircData(d, N, 'random');
filename_data = 'CyclicGroup6.txt';
X = genGroupData(filename_data, N);
P = importPermutationsToMat(filename_data);
n = size(X, 2);

%% unit-norm data
normx = sqrt(sum(X.^2, 1));
X = bsxfun(@times, X, 1./normx);

%% KSVD

% k = 50; % sparsity of each example
optKSVD.data = X;
optKSVD.Tdata = 1;
optKSVD.Edata = 10^-1;
optKSVD.dictsize = k;
optKSVD.iternum = 50;
optKSVD.memusage = 'high';
optKSVD.codemode = 'sparsity';

[D, A, err] = ksvd(optKSVD,'');

%% Signatue comparison
nTrials = 10; % number of random templates to use
[ds_in, ds_out, Wo] = ComparisonGD(filename_data, D, false, nTrials);
nBins = 10;
dispComparisonGD(ds_in, ds_out, nBins);

%% Images
figure;
subplot(2,2,1); imagesc(D); colorbar; title('Dictionary')
subplot(2,2,2); imagesc(A); colorbar; title('A')
subplot(2,2,4); imagesc(X); colorbar; title('Data')
subplot(2,2,3); imagesc(D'*D); colorbar; title('Gramian')


%% Quantitative Comparisons
disp(' ');
A = full(A);
fprintf('Reconstruction = %f\n', (1/n)*norm(X-D*A, 'fro').^2);
fprintf('Sparsity A, |A|_0 = %f, |A|_1 = %f\n', sum(A(:)==0)/numel(A), sum(abs(A(:)))/numel(A));

% Is resulting dictionary a group?
fprintf('Group, std(sum(D^T*D)) = (%f, %f)\n', std(sum(D'*D, 1)), std(sum(D'*D, 2)));




