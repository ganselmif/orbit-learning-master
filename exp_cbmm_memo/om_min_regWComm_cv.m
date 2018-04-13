% CrossValidating for selecting various lambdas
%
% reg(W) + Comm(XX,WW) + \sum(reg(ReLU(W)) + SC(W)
%
% Example:
% om_min_regWComm_cv(2, 6, 1, [], 0, 1, '/om/user/gevang/exp/orblearn/data')
%
% Taken from: exp_min_regWComm_num_training.m, script_min_regWComm.m

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function om_min_regWComm_cv(group_id, dim_group, lambda_n, lambda_w_vals, lambda_s_vals, sd, dirExp)

if nargin<7, dirExp = '/media/gevang/Data/work/exp/orblearn/data'; end % dir where mat data files are located
if nargin<6, sd = 1; end % seed/partition/random set
if nargin<2, dim_group = 6; end
if nargin==0, group_id = 1; end

%% ReLUs lambda
if nargin<3 || isempty(lambda_n), lambda_n = 1; end
%% Commutator lamba
if nargin<4 || isempty(lambda_w_vals), lambda_w_vals = 10.^(0:1:4); end
nLambdas = length(lambda_w_vals);
%% Self-Coherence lambda
if nargin<5 || isempty(lambda_s_vals), lambda_s_vals = 0; end % 10.^(-3:1:0);
nLambdas_s = length(lambda_s_vals);

%% Group names
switch dim_group
    case 12
        dataGroupName = {'Cyclic', 'Dicyclic'};
    case 6
        % dirData = sprintf('/media/gevang/Data/work/code/orblearn/datasets_groups/group_matrices/dim%d', dim_group);
        dataGroupName = {'Cyclic', 'Pyritohedral', 'Dihedral'};
    case 4
        % dirData = sprintf('/media/gevang/Data/work/code/orblearn/datasets_groups/group_matrices/dim%d', dim_group);
        dataGroupName = {'Alternating', 'Cyclic', 'Dihedral', 'Tetrahedral', 'Vierergruppe'};
end
tagData = dataGroupName{group_id};


%% Load data/get training set
filename_db = sprintf('imdb_%s_%d_sds_%d.mat', tagData, dim_group, 20);
load(fullfile(dirExp, filename_db));

% training set
Xr = X(:, ind_set==1, sd); % ind_xr = ind_orbit(ind_set==1);
[d, Nx] = size(Xr);
N = sum(ind_t==1);
kOrbitSize = Nx./N;

% shuffle
r = randperm(Nx);
Xr = Xr(:, r);

%% Cached computations
XXt = Xr*Xr'/Nx; % (Nx-1);
% XXt = normdotprod(X, X);

clear X Xr ind_orbit ind_set filename_db sd_vec s dataGroupName r;

%% Parameters
s = 0.1; % spread of Gaussian approximating the Dirac
if lambda_n==0 && lambda_s_vals==0
    tagLoss = 'regWComm';
elseif lambda_s_vals==0
    tagLoss = 'regWCommReLU_100';
elseif lamda_n_vals==0
    tagLoss = 'regWComm_SC';
else
    tagLoss = 'regWCommReLU_100_SC';
end
numStartPoints = 50;


%% Optimization
% optimType = 'fminuncms'; %'fminunc';
% Unconstraint/Multiple start points
ITER = 5000;
tol = 10^-6;
useParallel = true;

for ls = 1:nLambdas_s
    
    % Self-coherence constant
    lambda_s = lambda_s_vals(ls);
    
    for l = 1:nLambdas
        
        % Commutator constant
        lambda_w = lambda_w_vals(l);
        
        %% ************************************************************************
        %% Minimization with Quasi-Newton and line-search: See also: script_min_regW.m
        % costFunc = @(t)regWCommFuncGradVec(t, XXt, kOrbitSize, d, s, lambda_w);
        costFunc = @(t)regWCommReLUFuncGradVec(t, XXt, kOrbitSize, d, s, lambda_w, lambda_n, lambda_s);
        
        %%  Initialization
        % rng('shuffle', 'twister'); scurr = rng;
        sd_points = 0; rng(sd_points, 'twister'); sd_points = rng;
        W = randSampleVec(d, kOrbitSize, 'uball');
        % vecW = vec(W);
        
        if useParallel && isempty(gcp), parpool; end
        [~, multiMinStruct] = fminMsWrapper(costFunc, vec(W), numStartPoints, 'uball', useParallel, ITER, 'off', tol);
        if useParallel delete(gcp); end
        
        % Retrieve all points and solutions ordered by min f val
        for c = 1:numStartPoints
            matVecW(:, c) = multiMinStruct(c).X; % converged points/solutions
            %    matVecWo(:, c) = multiMinStruct(c).X0{1}; % initial random points
        end
        %    fval = [multiMinStruct(:).Fval]; % min fval
        
        %% Metrics/Regularizers
        We = reshape(matVecW, [d, kOrbitSize, numStartPoints]);
        % W = reshape(matVecWo, [d, k, numStartPoints]);
        
        We_lambda(:,:,:,l) = We;
    end
    
    We_lambda_s(:,:,:,:,ls) = We_lambda;
    
end

We_lambda(:,:,:,:) = squeeze(We_lambda_s);

% Save CV matrices
if lambda_s==0
    % no SC term
    if nLambdas ==1, 
        tagLambda = lambda_w;
    else tagLambda = nLambdas;
    end
    filename_cv = fullfile(dirExp, sprintf('cv_%s_%s_%d_sd_%d_lw_%d_startpoints_%d_sdpoints_%d.mat', tagLoss, tagData, dim_group, sd, tagLambda, numStartPoints, 0));
    save(filename_cv, 'We_lambda', 'lambda_w_vals', 'sd_points');
else
    % SC term
    filename_cv = fullfile(dirExp, sprintf('cv_%s_%s_%d_sd_%d_lw_%d_ls_%d_startpoints_%d_sdpoints_%d.mat', tagLoss, tagData, dim_group, sd, nLambdas, nLambdas_s, numStartPoints, 0));
    save(filename_cv, 'We_lambda', 'lambda_w_vals', 'lambda_s_vals', 'sd_points');
end

