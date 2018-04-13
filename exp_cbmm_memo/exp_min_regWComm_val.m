% Crossvalidating for selecting lambda
%
% Taken from: exp_min_regWComm_num_training.m, script_min_regWComm.m,

clear; close all;

%% Load data
dirExp = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located

%% Group names 
dim_group = 6; group_id = 3; 
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

filename_db = sprintf('imdb_%s_%d_sds_%d.mat', tagData, dim_group, 20);
load(fullfile(dirExp, filename_db));

% choose seed/partition/random set
sd = 1;

%% Train and validation sets
% T = T(:,:,sd); X = X(:,:,sd);

% training set
% Xr = X(:, ind_set==1, sd); ind_xr = ind_orbit(ind_set==1);
% N = sum(ind_t==1);
% [d, Nx] = size(Xr); kOrbitSize = Nx./N;

% validation and test set (same seed)
% Xt = X(:, ind_set==3, sd);
Xv = X(:, ind_set==2, sd); ind_xv = ind_orbit(ind_set==2);
[d, Nx] = size(Xv); kOrbitSize = Nx./sum(ind_t==2);

clear X ind_orbit filename_db sd_vec s dataGroupName r;

%% ReLUs lambda
lambda_n = 1;
%% Commutator lamba
lambda_w_vals = 10.^(0:1:4);
nLambdas = length(lambda_w_vals);
%% Self-Coherence lambda
lambda_s_vals = 0; % 10.^(-3:1:0);
nLambdas_s = length(lambda_s_vals);

if lambda_n==0 && lambda_s_vals==0
    tagLoss = 'regWComm';
elseif lambda_s_vals==0
    tagLoss = 'regWCommReLU_100';
elseif lamda_n_vals==0
    tagLoss = 'regWComm_SC';
else
    tagLoss = 'regWCommReLU_100_SC';
end

%% Cross-validation/computation on a grid 
% om_min_regWComm_cv(group_id, dim_group, lambda_n, lambda_w_vals, lambda_s_vals, sd)

%% Load pre-computed CV matrices
numStartPoints = 50;
if lambda_s_vals==0
    % no SC term
    filename_cv = fullfile(dirExp, sprintf('cv_%s_%s_%d_sd_%d_lw_%d_startpoints_%d_sdpoints_%d.mat', tagLoss, tagData, dim_group, sd, nLambdas, numStartPoints, 0));
else
    % SC term
    filename_cv = fullfile(dirExp, sprintf('cv_%s_%s_%d_sd_%d_lw_%d_ls_%d_startpoints_%d_sdpoints_%d.mat', tagLoss, tagData, dim_group, sd, nLambdas, nLambdas_s, numStartPoints, 0));
end
load(filename_cv)

%% ************************************************************************

%% Signature evaluation on validation set
% nDics = numStartPoints;
poolMethodNames = {'max', 'l2', 'lp', 'meanrelu', 'rms', 'mean', 'hist', 'moments', 'centmom', 'sumstats', 'cdf', 'histc'};
distMethodNames = {'euclidean', 'cosine', 'seuclidean', 'chebychev', 'minkowski', 'hamming'};

% Pool and distance method
poolMethod = poolMethodNames{2};
distMethod = distMethodNames{1};
typeSignature = 'multi';

%% Validation/trials set
Nv = sum(ind_t==2);
% nTrials = Nv % random templates to test

% util matrices
[ind_in, ind_out] = indMatInterIntraOrbit(Nv, kOrbitSize);
index_point = kron(1:numStartPoints, ones(1, kOrbitSize))'; % orbit index: for pooling
clear dist_in_w dist_out_w dist_in dist_out distMat featMat

for ls = 1:nLambdas_s
   
    for l = 1:nLambdas
        
        % make multicomponent signature from multiple start points
        D = reshape(We_lambda(:,:,:,l,ls), [d, kOrbitSize*numStartPoints]);
        
        switch typeSignature
            
            case 'multi'
                index_pool = kron(1:numStartPoints, ones(1, kOrbitSize))'; % orbit index: for pooling
                [dist_in_w(:,l), dist_out_w(:,l), distMat(:,:,l,ls), featMat(:,:,l,ls)] = makeSignatureDist(Xv, D, poolMethod, distMethod, index_pool, ind_in, ind_out);
                
            case 'single'
                indComp = 1; nComp = 1;
                Dt = D(:, index_point==indComp);
                index_pool = kron(1:nComp, ones(1, kOrbitSize))'; % orbit index: for pooling
                [dist_in_w(:,l,1), dist_out_w(:,l,1), distMat(:,:,l,ls), featMat(:,:,l,ls)] = makeSignatureDist(Xv, Dt, poolMethod, distMethod, index_pool, ind_in, ind_out);
        end
        
    end
    dist_in(:,:,ls) = dist_in_w; 
    dist_out(:,:,ls) = dist_out_w;     
end


% Signature distance
nBins = 50; ls = 1;
str_lambda = cellfun(@num2str, num2cell(lambda_w_vals), 'UniformOutput', false); % legend cell string
[stats_ds_in, stats_ds_out] = dispComparisonGD(dist_in(:,:,ls), dist_out(:,:,ls), nBins, str_lambda);


% for i=1:10
% [stats_ds_in, stats_ds_out] = dispComparisonGD(dist_in(:,:,1), dist_out(:,:,1), nBins, str_lambda);
% end

%% ************************************************************************
%% Stat plots
h1 = figure; semilogx(lambda_w_vals, stats_ds_in, 'o-', 'LineWidth', 2);
axis tight; ylabel('Intra-orbit'); xlabel('commutator reg. constant \lambda_w'); % title(typeData);
grid on;
legend('median', 'mean', 'std', 'location', 'best');

set(findall(gca, '-property', 'FontSize'), 'FontSize', 14)

h2 = figure; semilogx(lambda_w_vals, stats_ds_out, 'o-', 'LineWidth', 2);
axis tight; ylabel('Inter-orbit'); xlabel('commutator reg. constant \lambda_w');  % title(typeData);
grid on;
legend('median', 'mean', 'std', 'location', 'best');

set(findall(gca, '-property', 'FontSize'), 'FontSize', 14)


%% PRINT PLOTS
flag_print = true;

if flag_print
    figsDir = '/media/gevang/Data/work/exp/orblearn/figs';
    commonFileName =  sprintf('%s_%s_%s_StartPoints_%d_%s_%s', tagLoss, tagData, typeSignature, numStartPoints, poolMethod, distMethod);
    
    % printif(h1, fullfile(figsDir, sprintf('%s_%s_%s', 'lambdaw_cv', commonFileName, 'within')), flag_print, true, true);
    % printif(h2, fullfile(figsDir, sprintf('%s_%s_%s', 'lambdaw_cv', commonFileName, 'across')), flag_print, true, true);
        
    h = figure(1); grid off;
    set(h.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], 'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    % set(gcf, 'title', 'off');
    axis tight;
    set(findall(h, '-property', 'FontSize'), 'FontSize', 14)
    
    printif(h, fullfile(figsDir, sprintf('%s_%s_%s', 'lambdaw_dist', commonFileName, 'within')), flag_print, true, true);
    
    h = figure(2); grid off;
    set(h.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], 'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    % set(gcf, 'title', 'off');
    set(findall(h, '-property', 'FontSize'), 'FontSize', 14)
    
    printif(h, fullfile(figsDir, sprintf('%s_%s_%s', 'lambdaw_dist', commonFileName, 'across')), flag_print, true, true);
    
end


%% ************************************************************************
%% Analyzing the matrices on the validation set

lw = nLambdas; ls = 1; % picking largest lambda

% Signature distance
lambda_w = lambda_w_vals(lw);
distMatSigna = distMat(:,:,lw,ls); 
% Feature map
featMatSigna = featMat(:,:,lw,ls);

% Pixel distance
featMapPixel = Xv';
distMatPixel = pdist2(featMapPixel, featMapPixel, distMethod);

% Inter-intra- orbit distances
no_orb_show = 3;
[h1, ~] = imOrbitDistMatrix(distMatPixel, ind_in, ind_out, kOrbitSize, no_orb_show); close;
[h2, ~] = imOrbitDistMatrix(distMatSigna, ind_in, ind_out, kOrbitSize, no_orb_show); close;

h5 = plotDistMatrix(distMatPixel, ind_in, ind_out, 30);% nComp)
h6 = plotDistMatrix(distMatSigna, ind_in, ind_out, 30);% nComp)

%% Set same axis to both for direct comparisons
% x_lim = [max(h5.CurrentAxes.XLim, h6.CurrentAxes.XLim)]; 
% y_lim = max(h5.CurrentAxes.YLim, h6.CurrentAxes.YLim);
% set(h5.CurrentAxes, 'XLim', x_lim, 'YLim', y_lim);
% set(h6.CurrentAxes, 'XLim', x_lim, 'YLim', y_lim);

no_dims = 2;
typeVis = 'tSNE';

switch typeVis
    
    case 'mds'
        %% Classic MD scaling
        [visDistMatPixel, ep] = cmdscale(distMatPixel, no_dims);
        [visDistMatSigna, es] = cmdscale(distMatSigna, no_dims);
        
    otherwise
        %% t-SNE
        perplexity = 10; nTrials = 6;
        visDistMatPixel = mytsne(featMapPixel, 'tsne', nTrials, no_dims, perplexity);
        visDistMatSigna = mytsne(featMatSigna, 'tsne', nTrials, no_dims, perplexity);
        
end
clear featMapPixel featMapSign

% Visualization
fn = 14; markersize = 40;

n_orbit_to_show = 200;
ind_orbit_to_show = randperm(length(unique(ind_xv)), n_orbit_to_show) + min(ind_xv)-1;
ind_points_from_orbit = ismember(ind_xv', ind_orbit_to_show', 'rows');
sum(ind_points_from_orbit)

label_orbit_to_show = kron(1:n_orbit_to_show, ones(1, kOrbitSize));
cmap = lines(n_orbit_to_show);

h7 = figure;
scatter(visDistMatPixel(ind_points_from_orbit,1), visDistMatPixel(ind_points_from_orbit,2), markersize, label_orbit_to_show, 'filled'); colormap(cmap); axis equal;
set(gca, 'box', 'off', 'zticklabel',[], 'xticklabel',[], 'yticklabel',[], 'fontsize', fn, 'FontName', 'Arial')
% set(findall(h, '-property', 'FontSize'), 'FontSize', 14)

h8 = figure;
try
    scatter(visDistMatSigna(ind_points_from_orbit,1), visDistMatSigna(ind_points_from_orbit,2), markersize, label_orbit_to_show, 'filled');
catch
    scatter(visDistMatSigna(ind_points_from_orbit,1), zeros(size(visDistMatSigna)), markersize, label_orbit_to_show, 'filled');
end
colormap(cmap); axis equal;
set(gca, 'box', 'off', 'zticklabel',[], 'xticklabel',[], 'yticklabel',[], 'fontsize', fn, 'FontName', 'Arial')
% set(findall(h, '-property', 'FontSize'), 'FontSize', 14)


if flag_print
    commonFileName =  sprintf('%s_%s_%s_StartPoints_%d_%s_%s', tagLoss, tagData, typeSignature, numStartPoints, poolMethod, distMethod);
    
    set(h1.CurrentAxes, 'Visible','off'); figure(h1);
    printif(h1, fullfile(figsDir, sprintf('%s_%d_val_%s', 'distMat', no_orb_show, 'pixel')), flag_print, false, false);
    close(h1);
    
    set(h2.CurrentAxes, 'Visible','off'); figure(h2);
    printif(h2, fullfile(figsDir, sprintf('%s_%d_val_%s', 'distMat', no_orb_show, commonFileName)), flag_print, false, false);
    close(h2);
    
    figure(h5); grid off;
    set(h5.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], 'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    set(findall(h5, '-property', 'FontSize'), 'FontSize', 14)
    printif(h5, fullfile(figsDir, sprintf('%s_val_%s', 'distDist', 'pixel')), flag_print, true, true);
    close(h5);
    
    figure(h6); grid off;
    set(h6.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], 'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    set(findall(h5, '-property', 'FontSize'), 'FontSize', 14)
    printif(h6, fullfile(figsDir, sprintf('%s_val_%s', 'distDist', commonFileName)), flag_print, true, true);
    close(h6);
    
    figure(h7);
    printif(h7, fullfile(figsDir, sprintf('%s_%d_val_%s', typeVis, no_dims, 'pixel')), flag_print, true, true);
    close(h7);
    
    figure(h8);
    printif(h8, fullfile(figsDir, sprintf('%s_%d_val_%s', typeVis, no_dims, commonFileName)), flag_print, true, true);
    close(h8);
end




