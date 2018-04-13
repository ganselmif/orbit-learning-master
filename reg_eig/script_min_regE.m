% Script to minimize the reg(W) and regE from random initialization
% No other terms/constraints

% See also: script_min_regW.m script_min_regWX.m script_min_regWComm.m

clear; %close all;
debug = false; % flag for debugging

%% Parameters
s = 0.1;
lambda = 1; % eig reg weight

%% Data generation
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, W, k, d, filename_data] = gen_data_toy_orbits(typeData);
typeData = 'group'; N = 100; [X, ~, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
Nx = size(X, 2);

% Normalize by max eigenvalue
XXt = X*X';
XXt = XXt/max(eig(XXt));

% rng(0); % 'default');
% Initialize
W = rand([d, k]);  
W = project_pos(W); W = bsxfun(@rdivide, W, sqrt(sum(W.^2)));

%% ************************************************************************
if debug
    %% Test gradients    
    diff = testgradE(W, XXt, d, k, s);   
end
%% ************************************************************************

%% Auxiliary constants for gradient/regularizer 
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% ************************************************************************
%% Alternative minimization: See also: script_min_regW.m
% [J, gradW] = regWXFuncGradVec(vecW, X, k, d, s);
costFunc = @(t)(regWEFuncGradVec(t, XXt, k, d, s, lambda));
vecW = vec(W);

if 0    
    %% Check gradients (sanity check)
    [~, gradW] = costFunc(vecW);
    numgradW = computeNumericalGradient(costFunc, vecW);
    disp([numgradW gradW]);
    diff = norm(numgradW-gradW)/norm(numgradW+gradW);
    fprintf('W gradient: %e \n', diff);
end


%% Optimization
optimType = 'fmincon'; %'fminunc';
ITER = 5000;

switch optimType
    
    case {'fminunc', 'minFunc'}
        % Unconstraint
        tol = 10^-8;
        vecW = fminWrapper(costFunc, vecW, optimType, ITER, 'on', tol);
        
    case {'fminuncms'}
        % Unconstraint/Multiple start points
        tol = 10^-8;
        numStartPoints = 10;
        vecW = fminMsWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', tol);
        
    case 'fmincon'
        % Contraint optimization 
        numStartPoints = 1;
        
        %% Unit ball                
        % nonlcon = @(t)unitball(t, k); % unit ball contraint(s), i.e. sum of vector norms<k
        
        %% Unit norm constraints
        nonlcon = @(t)unitnorm(t, d, k); % unit norm contraint 
        
        %% Positivity contraints
        lb = zeros(size(vecW));
        
        condOpt{1} = nonlcon;
        condOpt{2} = lb;
        vecW = fminConWrapper(costFunc, vecW, numStartPoints, 'randb', true, ITER, 'on', condOpt);       
end


%% Metrics/Regularizers
We = reshape(vecW, d, k);
reg_Wo = regW_fixed(W, k, s, kE_term);     % original W
reg_W = regW_fixed(We, k, s, kE_term);     % converged W
reg_XW = regW_fixed(X'*We, k, s, kE_term); % X'W 
com_a = norm(comm(XXt/Nx, We*We'), 'fro'); % commutator norm
rege_Wo = regE(W, XXt, d, k, s); % Eigenvalue reg (original W)
rege_W = regE(We, XXt, d, k, s); % Eigenvalue reg (converged W)

fprintf('reg(Wo): %e, reg(W): %e, regE(Wo): %e, regE(W): %e, reg(X''W): %e, norm_com: %f\n', ...
    reg_Wo, reg_W, rege_Wo, rege_W, reg_XW, com_a)

% Plots
figure;
Ge = We'*We; GXe = We'*(XXt)*We;
subplot(1,3,1); imagesc(We); colorbar; title('Weights'); axis square
subplot(1,3,2); imagesc(Ge); colorbar; title('Gramian'); %axis equal
subplot(1,3,3); imagesc(GXe); colorbar; title('Gramian XtW'); %axis equal


fprintf('Ge, (med, mean, std) = (%f, %f, %f)\n', median(Ge(:)), mean(Ge(:)), std(Ge(:)));
fprintf('GXe, (med, mean, std) = (%f, %f, %f)\n', median(GXe(:)), mean(GXe(:)), std(GXe(:)));
