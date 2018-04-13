%% Logistic Regression

clear; % clear all;
close all hidden;

%% Data X (d x n)
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random

N = 6; % number of templates/untransformed data

% k = 6;   % dictionary size
% d = 6;   % data dimension and for now also data number...?
% c = linspace(0.1, 1, d); X = gallery('circul', c);
% X = genRandCircData(d, N, 'random');

% filename_group = 'DicyclicGroup7.txt';
filename_group = 'CyclicGroup6.txt';
[X, X0] = genGroupData(filename_group, N);
P = importPermutationsToMat(filename_group);

[d, n] = size(X);

% % % unit-norm data
% normx = sqrt(sum(X.^2, 1));
% X = bsxfun(@times, X, 1./normx);

%% Form supervised data matrices (X,y)
X = X';
y = kron(1:N, ones(1,d))'; % class labels
nClass = N;
classLabels = unique(y);

%% Visualize
Xu = project_pca(X, 2);
% classLabel = [1:N];
% classInd = [1:N];
myscatter(Xu, y); %, classInd); %,  %, fn) %, nDims) %, classColor)
% Labels and Legend
xlabel('pc 1')
ylabel('pc 2')

%==========================================================================
% Split train/test partitions

indTest = randsplitho(y, 0.2);
indTrain = setdiff(1:n, indTest)';

Xtr = X(indTrain,:); ytr = y(indTrain, :);
Xte = X(indTest, :); yte = y(indTest, :);

%==========================================================================

% Polynomial feature map term is handled % Xh = mapSecondOrderFeatures(X); Xh = Xh(:,2:end);
Xh = Xtr; yk = labels_multi_to_bin(ytr);

%% Regularized (Kernel) Logistic Regression
type_lr = 'klr';
% Regularization parameter (set to 1, or try different orders e.g. [0,1,10,100]
lambda = 1;
% cross-validate this and sigma 

switch type_lr
    case 'klr'
        % Kernel
        K = kernelrbf(Xh');
        costFunc = @(t)(costFunctionKerLogReg(t, K, yk, lambda));
        % Initialize fitting parameters
        C0 = randInitializeWeights(nClass, size(Xh, 1)); C0(:,1) = [];
        vec_W0 = C0(:); % unroll parameters
    otherwise
        costFunc = @(t)(costFunctionLogReg(t, Xh, yk, lambda));
        % Initialize fitting parameters % w0 = zeros(size(Xh, 2), 1);
        W0 = randInitializeWeights(size(Xh, 2), nClass);
        vec_W0 = W0(:); % unroll parameters
end

if 0
    % Check gradients
    [~, grad] = costFunc(vec_W0);
    numgrad = computeNumericalGradient(costFunc, vec_W0);
    % disp([numgrad grad]);
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    fprintf('W gradient: %e \n', diff);
end


%% Optimization
fprintf('Cost at initial w (zeros): %f\n', costFunc(vec_W0));

% options = optimset('GradObj', 'on', 'MaxIter', 400);
% [theta, J, exit_flag] = fminunc(costFunc, initial_theta, options);
optimType = 'fminunc';
max_iter = 100;
vec_W = fminWrapper(costFunc, vec_W0, optimType, max_iter);
% p = fminWrapper(costFunc, initial_theta, optimType, MAX_ITER)
fprintf('Cost at converged w: %f\n', costFunc(vec_W));


switch type_lr
    case 'klr'
        
        W = reshape(vec_W, [size(K, 2) size(yk, 2)]);
        
        %% Train set predicition/accuracy
        [K, s] = kernelrbf(Xh'); % Kernel
        pred_tr = klr_predict(W, K);
        %% Test set accuracy
        Kte = kernelrbf(Xh', s, Xte');
        pred_te = klr_predict(W, Kte');
    otherwise
        
        % Reshape W_vec back into the parameters W (cell array!)
        W = paramVec2Mats(vec_W, [size(Xh, 2) size(yk, 2)], 1); W = W{:};
        
        %% Train set predicition/accuracy
        pred_tr = lr_predict(W, Xh);
        %% Test set accuracy
        pred_te = lr_predict(W, Xte);
end

ytr_pred = classLabels(pred_tr);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(ytr_pred == ytr)) * 100);

yte_pred = classLabels(pred_te);
fprintf('\nTest Set Accuracy: %f\n', mean(double(yte_pred == yte)) * 100);

