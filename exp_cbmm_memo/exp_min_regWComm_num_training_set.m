% Experiments for minimizing reg(W) and comm(WWt, XXt)
%
% Taken from: script_min_regWComm.m

clear; close all;

%% Parameters
s = 0.1; % spread of Gaussian approximating the Dirac
lambda_w = 10; % regcomm weight

%% Data generation
sd = 666; 
rng(sd, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, k, d, filename_data, W] = gen_data_toy_orbits(typeData);
typeData = 'DihedralGroup6';

Nvals = 10.^(0:5);

for n = 1:length(Nvals)
    
    N = Nvals(n); % [X, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
    filename_data = [typeData '.txt']; X = genGroupData(filename_data, N); [d, Nx] = size(X); k = Nx./N;
    % X = X(:, randi1:end);
    % X = X(:,randi(size(X,2), [1,1000]));
    
    %% Auxiliary constants for gradient/regularizer
    E = kron(eye(k), ones(k));
    Ik = sparse(eye(k));
    kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
    [C, R] = gradW_opt_aux(k);
    CRt = R'*C';
    
    %% Cached computations
    % XW = X'*W;
    XXt = X*X'/Nx; % (Nx-1);
    % XXt = normdotprod(X, X);
    
    %% ************************************************************************
    %% Minimization with Quasi-Newton and line-search: See also: script_min_regW.m
    
    %% Compact constraint
    costFunc = @(t)regWCommFuncGradVec(t, XXt, k, d, s, lambda_w);
    %% Compact + all columns contraints
    % costFunc = @(t)regWCommColumnsFuncGradVec(t, XXt, k, d, s, lambda_w);
    %% Compact constraint + Log(Det(Cov)))
    % lambda_c = 0.001;
    % costFunc = @(t)regWCommDetCovFuncGradVec(t, XXt, k, d, s, lambda_w, lambda_c);
    
    %%  Initialization
    rng('shuffle', 'twister'); scurr = rng;
    % rng(0, 'twister'); scurr = rng;
    W = randSampleVec(d, k, 'uball');
    % vecW = vec(W);
    
    %% Optimization
    % optimType = 'fminuncms'; %'fminunc';
    ITER = 5000;
    
    % case {'fminuncms'}
    % Unconstraint/Multiple start points
    tol = 10^-8;
    numStartPoints = 10;
    
    if isempty(gcp), parpool; end
    [vecW, multiMinStruct] = fminMsWrapper(costFunc, vec(W), numStartPoints, 'uball', true, ITER, 'off', tol);
    delete(gcp);
    
    % Retrieve all points and solutions ordered by min f val
    for c = 1:numStartPoints
        matVecW(:, c) = multiMinStruct(c).X; % converged points/solutions
        % matVecWo(:, c) = multiMinStruct(c).X0{1}; % initial random points        
    end
    % fval = [multiMinStruct(:).Fval]; % min fval
    
    %% Metrics/Regularizers
    We = reshape(matVecW, [d, k, numStartPoints]);
    % W = reshape(matVecWo, [d, k, numStartPoints]);
    
    %% Invariance Test/Comparison signatures
    nTrials = 500; % random templates to test
    nBins = 50;
    
    [ds_in, ds_out, Wo] = ComparisonGD(filename_data, We, false, nTrials); %,'CyclicGroup6.txt');
    [stats_ds_in(:,:,n), stats_ds_out(:,:,n)] = dispComparisonGD(ds_in, ds_out, nBins);    
end

%% Learned algorithm stats
regw_stats_in = squeeze(stats_ds_in(:,2,:));
regw_stats_out = squeeze(stats_ds_out(:,2,:));

h1 = figure; semilogx(Nvals, regw_stats_in, 'o-', 'LineWidth', 2);
axis tight; ylabel('Intra-orbit'); xlabel('training set size'); title(typeData); 
grid on;
legend('median', 'mean', 'std', 'location', 'best'); 

set(findall(gca, '-property', 'FontSize'), 'FontSize', 14)

h2 = figure; semilogx(Nvals, regw_stats_out, 'o-', 'LineWidth', 2);
axis tight; ylabel('Inter-orbit'); xlabel('training set size'); title(typeData); 
grid on;
legend('median', 'mean', 'std', 'location', 'best'); 

set(findall(gca, '-property', 'FontSize'), 'FontSize', 14)


%% PRINT PLOTS
flag_print = true;

if flag_print
    tagGroup = typeData;
    figsDir = '/media/gevang/Data/work/exp/orblearn/figs';
    commonFileName = fullfile(figsDir, sprintf('regw_dep_N_sd_%d_%s_l_%d_nS_%d', sd, tagGroup, lambda_w, numStartPoints));
     
    printif(h1, sprintf('%s_%s', commonFileName, 'within'), flag_print, true, true);
    printif(h2, sprintf('%s_%s', commonFileName, 'across'), flag_print, true, true);
end

