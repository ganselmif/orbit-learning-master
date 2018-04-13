% Script to minimize reg(W) and comm(WWt, XXt) from random initialization
% No other terms/constraints
%
% See also: script_min_regW.m script_min_regWX.m
%
% Renamed/changed from main_algo_regW.m

clear; close all;
debug = false; % flag for debugging

%% Parameters
s = 0.1; % spread of Gaussian approximating the Dirac
lambda_w = 10^4; % comm weight
lambda_n = 1; % mReLU weight
lambda_s = 0.001;% SC weight

%% Data generation
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, k, d, filename_data, ~, W] = gen_data_toy_orbits(typeData);
typeData = 'group'; N = 1000; [X, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
% X = X(:, randi1:end);
% X = X(:,randi(size(X,2), [1,1000]));
Nx = size(X, 2);

% Initialization
rng('shuffle', 'twister') 
W = randSampleVec(d, k, 'uball');

%% ************************************************************************
% Test gradient of commutator function
if debug
    % Test gradient function
    diff1 = testgradWComm(W);
    diff2 = testgradWComm(W, X);
    
    % With all columns contraint
    % diff = testgradWCommColumns(W);
    % diff = testgradWCommColumns(W, X);    
end
% ************************************************************************

%% Cached computations
XXt = X*X'/Nx; % (Nx-1);
% XXt = normdotprod(X, X);

% %% Regular/Naive Gradient Descend
% if False
%     
%     % Auxiliary constants for gradient/regularizer
%     E = kron(eye(k), ones(k));
%     Ik = sparse(eye(k));
%     kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
%     [C, R] = gradW_opt_aux(k);
%     CRt = R'*C';
%     
%     ITER = 1000; % num of iterations
%     tau = 0.5;  % learning rate
%     beta = 0.1; % decrease in learning step size ([0.1, 0.8])
%     p = s; % 0.1;
%     
%     for iterNo = 1 : ITER
%         Wo = W;
%         
%         WWt = W*W';
%         % WWt = normdorprod(W, W);
%         
%         % grad_W1 = 4*comm(comm(WWt, XXt), XXt)*W;
%         grad_W1 = 4*comm2(WWt, XXt)*W; % maybe a bit faster
%         
%         % grad_W2 = reshape(gradW_opt_1(W, k, p),[d,k]);
%         grad_W2 = reshape(gradW_opt_1_fixed(W, k, p, Ik, E, CRt), [d, k]);
%         
%         nder1(iterNo) = norm(grad_W1(:));
%         nder2(iterNo) = norm(grad_W2(:));
%         
%         W = W - tau*(lambda_w*grad_W1 + grad_W2);
%         % W  = project_unit(W);
%         
%         reg1(iterNo) = regW_fixed(W, k, s, kE_term);
%         reg2(iterNo) = regW_fixed(X'*W, k, s, kE_term);
%         com(iterNo) = norm(comm(XXt, W*W'), 'fro');
%         dW(iterNo)= norm(Wo - W, 'fro'); % convergence track
%         
%         fprintf('%4d, reg(W): %f, reg(X''W): %f, norm_com: %f, dW: %f\n', ...
%             iterNo, reg1(iterNo), reg2(iterNo), com(iterNo), dW(iterNo))
%         
%         tau = beta*tau;
%     end
%     
%     %% Plots and Figures
%     figure;
%     r1 = reg1(1)/reg1(end);
%     r2 = reg2(1)/reg2(end);
%     subplot(2,3,1); imagesc(W); colorbar; title('Weights');
%     subplot(2,3,2); imagesc(W'*W); colorbar; title('Gramian(W)');
%     subplot(2,3,3); imagesc((X'*W)'*(X'*W)); colorbar; title('Gramian(XtW)')
%     %figure;
%     subplot(2,4,5); semilogx(1:ITER, reg1(1:end), '.-'); axis tight; title(sprintf('rW1/rend = %0.2e ',r1));
%     subplot(2,4,6); semilogx(1:ITER, reg2(1:end), '.-'); axis tight; title(sprintf('rW2/rend = %0.2e ',r2));
%     subplot(2,4,7); semilogx(1:ITER, nder1(1:end), '.-'); axis tight; title('norm dW1');
%     subplot(2,4,8); semilogx(1:ITER, nder2(1:end), '.-'); axis tight; title('norm dW2');
%     %subplot(4,4,8); plot(1:ITER, com(1:end), '.-'); axis tight; title('norm Comm');
%     
%     figure; imagesc(comm(X*X',W*W'));colorbar;
%     
% end

%% ************************************************************************
%% Minimization with Quasi-Newton and line-search: See also: script_min_regW.m

%% Compact constraint
% costFunc = @(t)regWCommFuncGradVec(t, XXt, k, d, s, lambda_w); 
%% Compact constraint + all columns 
% costFunc = @(t)regWCommColumnsFuncGradVec(t, XXt, k, d, s, lambda_w);
%% Compact constraint + Log(Det(Cov)))
% lambda_c = 0.001;
% costFunc = @(t)regWCommDetCovFuncGradVec(t, XXt, k, d, s, lambda_w, lambda_c);
%% Compact constraint + multiple ReLUs (nonlin Grammian) 
costFunc = @(t)regWCommReLUFuncGradVec(t, XXt, k, d, s, lambda_w, lambda_n, lambda_s); 

vecW = vec(W);

if debug    
    %% Check gradients (sanity check)
    [~, gradW] = costFunc(vecW);
    numgradW = computeNumericalGradient(costFunc, vecW);
    disp([numgradW gradW]);
    diff = norm(numgradW-gradW)/norm(numgradW+gradW);
    fprintf('W gradient: %e \n', diff);
end


%% Optimization
optimType = 'fminunc'; %'fminunc';
ITER = 5000;

switch optimType
    case 'fmin_adam'
        
        opt = optimset('fmin_adam');
        opt.GradObj = 'on';
        opt.MaxFunEvals = 1e4;        
        opt.Display = 'on';
        opt.MaxIter = ITER;
        opt.TolFun = 1e-8;
        opt.TolX = 1e-8;
        
        [vecW, fval, ~, ~] = fmin_adam(costFunc, vecW, 0.01, 0.9, 0.999, sqrt(eps), [], opt); %stepSize, beta1, beta2, epsilon, nEpochSize, options);
    
    case {'fminunc', 'minFunc'}
        % Unconstraint
        tol = 10^-8;
        % vecW = fminWrapper(costFunc, vecW, optimType, 1, 'on', tol);
        vecW = fminWrapper(costFunc, vecW, optimType, ITER, 'on', tol);
        
    case {'fminuncms'}
        % Unconstraint/Multiple start points
        tol = 10^-8;
        numStartPoints = 10;
        
        if isempty(gcp), parpool; end
        [vecW, multiMinStruct] = fminMsWrapper(costFunc, vecW, numStartPoints, 'uball', true, ITER, 'off', tol);
        delete(gcp);
        
%         % Retrieve all points and solutions ordered by min f val
%         for c = 1:numStartPoints
%             matVecW(:, c) = multiMinStruct(c).X; % converged points/solutions
%             matVecWo(:, c) = multiMinStruct(c).X0{1}; % initial random points
%             fval = [multiMinStruct(:).Fval]; % min fval
%         end
        
    case 'fmincon'
        % Contraint optimization 
        numStartPoints = 1;
     
        % conType = 'nonlcon';
        % numStartPoints = 10;
        % nonlcon = @(t)unitball(t, k); % unit ball contraint(s), i.e. sum of vector norms<k       
        % vecW = fminConWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', nonlcon);       
        
        %% Unit ball                
        nonlcon = @(t)unitball(t, k); % unit ball contraint(s), i.e. sum of vector norms<k
        condOpt = nonlcon;
        
        %% Unit norm constraints
        % nonlcon = @(t)unitnorm(t, d, k); % unit norm contraint 
        
        %% Positivity contraints
        % lb = zeros(size(vecW));
        
        % condOpt{1} = nonlcon;
        % condOpt{2} = lb;
        vecW = fminConWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', condOpt);  
        
end


%% Metrics/Regularizers
We = reshape(vecW, d, k);

reg_Wo = regW_fixed(W(:,:,1), k, s);     % original W
reg_W = regW_fixed(We(:,:,1), k, s);     % converged W
reg_XW = regW_fixed(X'*We(:,:,1), k, s); % X'W
com_Wo = norm(comm(XXt, W(:,:,1)*W(:,:,1)'), 'fro'); % commutator norm
com_W = norm(comm(XXt, We(:,:,1)*We(:,:,1)'), 'fro'); % commutator norm
sc_Wo = sum(sum((normdotprod(W, W) - eye(k)).^2)); % self-coherence/framepotential
sc_W = sum(sum((normdotprod(We, We) - eye(k)).^2));

% det_a = log(det(We(:,:,c)*We(:,:,c)'))/d;
% rege_Wo = []; % regE(W, XXt, d, k, s); % Eigenvalue reg (original W)
% rege_W = []; % regE(We, XXt, d, k, s); % Eigenvalue reg (converged W)

% fprintf('reg(Wo): %e, reg(W): %e, regE(Wo): %e, regE(W): %e, reg(X''W): %e, norm_com(Wo): %e, norm_com(W): %e, det_cov: %f\n', ...
%    reg_Wo, reg_W, rege_Wo, rege_W, reg_XW, com_Wo, com_W, det_a)
fprintf('reg(Wo): %e, reg(W): %e, reg(X''W): %e, norm_com(Wo): %e, norm_com(W): %e, sc(Wo): %e, sc(W): %e\n', ...
    reg_Wo, reg_W, reg_XW, com_Wo, com_W, sc_Wo, sc_W)


%% Plots
figure;
Ge = We'*We; GXe = We'*(XXt)*We;
subplot(1,3,1); imagesc(We); colorbar; title('Weights'); axis square
subplot(1,3,2); imagesc(Ge); colorbar; title('Gramian'); axis square
subplot(1,3,3); imagesc(GXe); colorbar; title('Gramian XtW'); axis square 

fprintf('Ge  (med, mean, std) = (%f, %f, %f)\n', median(Ge(:)), mean(Ge(:)), std(Ge(:)));
fprintf('GXe (med, mean, std) = (%f, %f, %f)\n', median(GXe(:)), mean(GXe(:)), std(GXe(:)));


if strcmp(typeData, 'group')
    %% Invariance Test/Comparison signatures
    nTrials = 500; % random templates to test
    
    [ds_in, ds_out, Wo] = ComparisonGD(filename_data, We, false, nTrials);
   
    nBins = 40; 
    dispComparisonGD(ds_in, ds_out, nBins);    

    if 0
        % True orbit plots
        figure;
        Wo = Wo(:, randperm(k)); Go = Wo'*Wo; GXo = Wo'*(XXt)*Wo;
        subplot(1,3,1); imagesc(Wo); colorbar; title('Orbit Weights'); axis square
        subplot(1,3,2); imagesc(Go); colorbar; title('Orbit Gramian'); axis square
        subplot(1,3,3); imagesc(GXo); colorbar; title('Gramian XtW'); axis square
        
        fprintf('Go, (med, mean, std) = (%f, %f, %f)\n', median(Go(:)), mean(Go(:)), std(Go(:)));
        fprintf('GXo, (med, mean, std) = (%f, %f, %f)\n', median(GXo(:)), mean(GXo(:)), std(GXo(:)));
    end
    
elseif strcmp(typeData, 'irotmnist')
    figure; display_network(X, false, true); % all instances/orbits from class c
    figure; display_network(We, false, true); % all instances/orbits from class c
end

