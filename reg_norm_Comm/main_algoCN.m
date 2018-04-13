%
% Renamed/changed from Fabio's main_algoCN.m


clear; close all;
debug = false; % flag for debugging

filename_group = 'CyclicGroup6.txt'; 
%filename_group = 'DihedralGroup15.txt'; 
% filename_group = 'DihedralGroup6.txt'; 
% filename_group = 'DicyclicGroup7.txt'; 
% filename_group = 'CrystallographicPointGroup17.txt';


%% Data generation
% rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random

d = 60;
k = 6; 
% W = genGroupData(filename_group, 1); % + 1*randn([d, k]);
% W = randArrayInRange([d, k, 1], 0, 1);
W = rand([d, k]);


%% Auxiliary functions
Jk = ones(k);
Jd = ones(d);
Id = eye(k);
k_term = k*Id-Jk;

%% Auxiliary functions
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';


%Dictionary (d x k)
%W = DatagenS(groupnametxt,1);
%W = randArrayInRange([d, k, 1], a, b);
%W = project_unit(project_pos(W)); % project in unit ball and positive


% Parameters
ITER = 1000;   % passes in ALS iterations
maxp = 3;
s = 0.1;
lambda_cn = 0.01;

%% ************************************************************************
% Test gradient 
if debug
    % Test gradient function
    diff = testgradWCN(W);
end
% ************************************************************************

if 0
    % Objective function
    f(1) = regCN(W, maxp, d, k); % LearnObjectiveFunctionCN(W, d, k, maxp);
    
    for iterNo = 1 : ITER
        Wo = W;
        
        grad_W1 = gradWCN(W, maxp, d, k);
        grad_W2 = reshape(gradW_opt_1_fixed(W, k, s, Ik, E, CRt), [d, k]);
        
        W = W - lambda_cn*grad_W1 - grad_W2;
        
        f(iterNo+1) = regCN(W, maxp, d, k); % LearnObjectiveFunctionCN(W, d, k, maxp);
        reCN(iterNo) = regCN(W, maxp, d, k);
        reW(iterNo)  = regW_fixed(W, k, s, kE_term);
        gradW1(iterNo) = norm(grad_W1, 'fro');
        gradW2(iterNo) = norm(grad_W2, 'fro');
        dW(iterNo)= norm(Wo - W, 'fro'); % convergence track
        
        for p=1:9
            com(p, iterNo) = (2/trace((W.^p)'*(W.^p)))*trace((W.^p)*(k_term)*(W.^p)'*Jd);
        end
        
        fprintf('%4d, f: %f, reCN: %f, reW: %f, dW: %f\n', iterNo, f(iterNo), reCN(iterNo), reW(iterNo), dW(iterNo));
        
    end
    
    
    %% Plots and Figures
    
    ITER_MAX_ALS = ITER;
    figure;
    subplot 231; plot(1:ITER_MAX_ALS, f(2:end), '.-'); axis tight; title('F');
    subplot 232; plot(1:ITER_MAX_ALS, reCN,'.-'); axis tight; title('comm')
    subplot 233; plot(1:ITER_MAX_ALS, reW, '.-'); axis tight; title('regW')
    subplot 234; plot(1:ITER_MAX_ALS, gradW1, '.-'); axis tight; title('gradW1')
    subplot 236; plot(1:ITER_MAX_ALS, gradW2, '.-'); axis tight; title('gradW2')
    
    
    figure
    for p=1:9
        subplot(3,3,p);
        plot(1:ITER_MAX_ALS, com(p,:), '.-'); axis tight; title(sprintf('com%d', p))
    end
    
    
    figure;
    % Dictionary
    subplot(2,1,1); imagesc(W); colorbar; title('Weights')
    % Gramian
    subplot(2,1,2); imagesc(W'*W); colorbar; title('Gramian(W)')
end


%% ************************************************************************
%% Minimization with Quasi-Newton and line-search: See also: script_min_regW.m, script_min_regWComm.m
costFunc = @(t)regCNFuncGradVec(t, maxp, k, d, s, lambda_cn);
vecW = vec(W);

if debug    
    %% Check gradients (sanity check)
    [~, gradW] = costFunc(vecW);
    numgradW = computeNumericalGradient(costFunc, vecW);
    disp([numgradW gradW]);
    diff = norm(numgradW-gradW)/norm(numgradW+gradW);
    fprintf('W gradient: %e \n', diff);
end


optimType = 'fminunc';
ITER = 5000;
tol = 10^-16;
vecW = fminWrapper(costFunc, vecW, optimType, ITER, 'on', tol);

% Metrics/Regularizers
We = reshape(vecW, d, k);
regW_fixed(W, k, s, kE_term);

reg1_a = regCN(We, maxp, d, k);
reg2_a = regW_fixed(We, k, s, kE_term);

disp('=====');
fprintf('Initial, reCN: %f, reW: %f\n', regCN(W, maxp, d, k), regW_fixed(W, k, s, kE_term));
fprintf('Minimiz, reCN: %f, reW: %f\n', reg1_a, reg2_a);
disp('=====');

% Plots
figure;
subplot(2,1,1); imagesc(We); colorbar; title('Weights');
subplot(2,1,2); imagesc(We'*We); colorbar; title('Gramian');

