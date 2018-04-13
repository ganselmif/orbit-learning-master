% Script to minimize reg(W) without other terms/constraints
%
% See also: script_min_regWComm.m script_min_regWX.m

clear; close all;
debug = false; % flag for debugging

%% Data generation
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, W, k, d, filename_data] = gen_data_toy_orbits(typeData);
typeData = 'group'; N = 1; [X, ~, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
Nx = size(X, 2);

% rng(0); % 'default');
typeInit = 'orbit_noise';
switch typeInit
    case 'orbit_noise'
        W = X + 0.1*randn([d, Nx]);
        
    case 'orbit_noise_location'
        W = X;  
        n = floor(0.1*numel(W)); % number of points to 'kill'         
        W(randi(numel(W),[n,1])) = mean(W(:));   
        
    case 'orbit'
        %% Normalized orbit
        W = bsxfun(@rdivide, X, sqrt(sum(X.^2)));
        
    otherwise
        %% Just noise
        % W = mean(X(:)) + std(X(:))*randn([d, Nx]);
        W = rand([d, Nx]);
        W = bsxfun(@rdivide, W, sqrt(sum(W.^2)));
end


%% Auxiliary constants for gradient/regularizer
E = kron(eye(k), ones(k));
% Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
% [C, R] = gradW_opt_aux(k);
% CRt = R'*C';

%% Cached computations
% XXt = X*X'/Nx; % (Nx-1);

%% Optimization
s = 0.1;
costFunc = @(t)(regWFuncGradVec(t, k, s));

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
    
    case {'fminunc', 'minFunc'}
        % Unconstraint
        tol = 10^-16;
        vecW = fminWrapper(costFunc, vecW, optimType, ITER, 'on', tol);
        
    case {'fminuncms'}
        % Unconstraint/Multiple start points
        tol = 10^-16;
        numStartPoints = 10;
        vecW = fminMsWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', tol);
        
    case 'fmincon'
        % Constraint/unit ball
        conType = 'nonlcon';
        numStartPoints = 10;
        nonlcon = @(t)unitball(t, k); % unit ball contraint(s), i.e. sum of vector norms<k
        vecW = fminConWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', nonlcon);  
end


% Metrics/Regularizers
We = reshape(vecW, d, k);
reg_W = regW_fixed(We, k, s, kE_term);
reg_X = regW_fixed(X, k, s, kE_term);
reg_Wo = regW_fixed(W, k, s, kE_term);

fprintf('reg(X): %e, reg(Wo): %e, reg(W): %e\n', reg_X, reg_Wo, reg_W)

% Plots
figure(2);
Ge = We'*We; % GXe = We'*(XXt)*We;
subplot(2,2,1); imagesc(We); colorbar; title('Weights'); %axis equal
subplot(2,2,2); imagesc(Ge); colorbar; title('Gramian'); %axis equal
% subplot(1,3,3); imagesc(GXe); colorbar; title('Gramian XtW'); %axis equal

fprintf('Ge, (med, mean, std) = (%f, %f, %f)\n', median(Ge(:)), mean(Ge(:)), std(Ge(:)));
% fprintf('GXe, (med, mean, std) = (%f, %f, %f)\n', median(GXe(:)), mean(GXe(:)), std(GXe(:)));


% True orbit plots
Wo = X; Go = Wo'*Wo; % GXo = Wo'*(XXt)*Wo;
subplot(2,2,3); imagesc(Wo); colorbar; title('Orbit Weights'); %axis equal
subplot(2,2,4); imagesc(Go); colorbar; title('Orbit Gramian'); %axis equal
% subplot(1,3,3); imagesc(GXo); colorbar; title('Gramian XtW'); %axis equal

fprintf('Go, (med, mean, std) = (%f, %f, %f)\n', median(Go(:)), mean(Go(:)), std(Go(:)));
% fprintf('GXo, (med, mean, std) = (%f, %f, %f)\n', median(GXo(:)), mean(GXo(:)), std(GXo(:)));


if strcmp(typeData, 'group')
    %% Invariance Test/Comparison signatures
    nTrials = 100; % number of random templates to use
    [ds_in, ds_out, Wo] = ComparisonGD(filename_data, We, false, nTrials);
    nBins = 10; 
    dispComparisonGD(ds_in, ds_out, nBins);    
end


%% Using NonZero term also (old/outdated)
% lambda1 = 1; lambda2 = 1;
% costFunc = @(t)(regWnzFuncGradVec(t, k, lambda1, lambda2, s1, s2));

% s = [0.001, 0.01, 0.1, 1, 10, 100];
%
% for i=1:6
%     for j=1:6
%         vecW = Wn(:);
%         lambda1 = 1; lambda2 = 1;
%         s1 = s(i); % 0.01;
%         s2 = s(j); % 0.01;
%         costFunc = @(t)(regWFuncGradVec(t, k, lambda1, lambda2, s1, s2));
%
%         optimType = 'fminunc';% 'minFunc';
%         vecWe = fminWrapper(costFunc, vecW, optimType, 1000, 'on');
%         We = reshape(vecWe, d, K);
%
%         J(i,j) = costFunc(vecWe);
%     end
% end



