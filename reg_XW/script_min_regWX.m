% Script to minimize the reg(W) and reg(XW) from random initialization
% No other terms/constraints
% 
% See also: script_min_regW.m
%
% Renamed/changed from Fabio's main_algo_regW.m

clear; %close all;
debug = false; % flag for debugging

%% Parameters
s = 0.1;
p = s; % 0.1;
lambda_w = 1; % regcomm weight

%% Data generation
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, W, k, d, filename_data] = gen_data_toy_orbits(typeData);
typeData = 'group'; N = 10; [X, W, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
Nx = size(X, 2);

% Auxiliary variables
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% ************************************************************************
if debug
    %% Test gradients
    diff = testgradWX;
    diff = testgradWX(W, X, d, k, 0.1);
    
    %% Sanity checks
    regW_fixed(W, k, s, kE_term) % random
    regW_fixed(X'*W, k, s, kE_term) % random WX
    
    regW_fixed(gen_data_toy_orbits(typeData, 1), k, s, kE_term)
    regW_fixed(X'*gen_data_toy_orbits(typeData, 1), k, s, kE_term)
    
    dW1 = gradWX_opt_1_fixed(W, X, k, s, Ik, E, CRt);
    dW2 = kron(Ik, X)*gradW_opt_1_fixed(X'*W, k, s, Ik, E, CRt);
    norm(dW1-dW2)
end
%% ************************************************************************

% if 0
%     % s_vals = [0.01 0.1 1:1:10 20 30];
%     s = 0.1;
%     ITER = 1000; % num of iterations
%     lambda = 1;
%     
%     normfro = @(p) sqrt(sum(sum(p.^2))); % Frobenius norm, fast
%     for iterNo = 1 : ITER
%         Wo = W;
%         %grad_W = reshape(gradW_opt_1_fixed(W, k, s),[d,k]) + reshape(gradWX_opt_1_fixed(W, X, k, s),[d,k]);
%         % grad_W = reshape(gradWX_opt_1_fixed(W, X, k, s), [d, k]);
%         grad_W = reshape(gradWX_opt_1_fixed(W, X, k, s), [d, k]);
%         
%         W = W - lambda*grad_W;
%         
%         %reg(iterNo)= regW(W, k, s) + regW(X'*W, k, s);
%         reg1(iterNo) = regW_fixed(X'*W, k, s, kE_term);
%         reg2(iterNo) = regW_fixed(W, k, s, kE_term);
%         nder(iterNo) = normfro(grad_W);
%         
%         %% convergence track
%         dW(iterNo)= normfro(Wo - W);
%         
%         fprintf('%4d, R(XtW): %f, R(W): %f, gW: %f, dW: %f\n', ...
%             iterNo, reg1(iterNo), reg2(iterNo), nder(iterNo), dW(iterNo))
%         
%     end
%     
%     %% Plots and Figures
%     r1 = reg1(1)/reg1(end);
%     r2 = reg2(1)/reg2(end);
%     
%     figure;
%     subplot(2,3,1); imagesc(W); colorbar; title('Weights');
%     subplot(2,3,2); imagesc(W'*W); colorbar; title('Gramian');
%     subplot(2,3,3); imagesc((X'*W)'*(X'*W)); colorbar; title('Gramian XtW')
%     subplot(2,3,4); plot(1:ITER, reg1(1:end), '.-'); axis tight; title(sprintf('rWX/rend = %0.2e ',r1));
%     subplot(2,3,5); plot(1:ITER, reg2(1:end), '.-'); axis tight; title(sprintf('rW/rend = %0.2e ',r2));
%     subplot(2,3,6); plot(1:ITER, nder(1:end), '.-'); axis tight; title('norm dW');
% end

%% ************************************************************************
%% Alternative minimization: See also: script_min_regW_only.m
% [J, gradW] = regWXFuncGradVec(vecW, X, k, d, s);
costFunc = @(t)(regWXFuncGradVec(t, X, k, d, s, lambda_w));
vecW = vec(W);

if 0    
    %% Check gradients (sanity check)
    [~, gradW] = costFunc(vecW);
    numgradW = computeNumericalGradient(costFunc, vecW);
    disp([numgradW gradW]);
    diff = norm(numgradW-gradW)/norm(numgradW+gradW);
    fprintf('W gradient: %e \n', diff);
end


optimType = 'fminunc';
ITER = 5000;
vecW = fminWrapper(costFunc, vecW, optimType, ITER, 'on');

% Metrics/Regularizers
We = reshape(vecW, d, k);
regW_fixed(W, k, s, kE_term)
reg1_a = regW_fixed(We, k, s, kE_term);
reg2_a = regW_fixed(X'*We, k, s, kE_term);
XXt = X*X'/Nx; % (Nx-1);
com_a = norm(comm(XXt, We*We'), 'fro');

fprintf('reg(W): %e, reg(X''W): %e, norm_com: %f\n', reg1_a, reg2_a, com_a)


figure;
subplot(1,3,1); imagesc(We); colorbar; title('Weights');
subplot(1,3,2); imagesc(We'*We); colorbar; title('Gramian');
subplot(1,3,3); imagesc(We'*(X*X')*We); colorbar; title('Gramian XtW')

fprintf('Ge, (med, mean, std) = (%f, %f, %f)\n', median(Ge(:)), mean(Ge(:)), std(Ge(:)));
fprintf('GXe, (med, mean, std) = (%f, %f, %f)\n', median(GXe(:)), mean(GXe(:)), std(GXe(:)));


if strcmp(typeData, 'irotmnist')
    figure; display_network(X, false, true); % all instances/orbits from class c
    figure; display_network(We, false, true); % all instances/orbits from class c
end

if strcmp(typeData, 'group')
    %% Invariance Test/Comparison signatures
    nTrials = 100; % number of random templates to use
    [ds_in, ds_out, Wo] = ComparisonGD(filename_data, We, false, nTrials);
    nBins = 10; 
    dispComparisonGD(ds_in, ds_out, nBins);        
   
    % True orbit plots
    figure;
    Wo = Wo(:, randperm(k)); Go = Wo'*Wo; GXo = Wo'*(XXt)*Wo;
    subplot(1,3,1); imagesc(Wo); colorbar; title('Orbit Weights'); %axis equal
    subplot(1,3,2); imagesc(Go); colorbar; title('Orbit Gramian'); %axis equal
    subplot(1,3,3); imagesc(GXo); colorbar; title('Gramian XtW'); %axis equal
    
    fprintf('Go, (med, mean, std) = (%f, %f, %f)\n', median(Go(:)), mean(Go(:)), std(Go(:)));
    fprintf('GXo, (med, mean, std) = (%f, %f, %f)\n', median(GXo(:)), mean(GXo(:)), std(GXo(:)));
        
end