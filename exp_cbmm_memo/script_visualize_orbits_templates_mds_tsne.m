% CBMM Memo, Figure 2
% 
% See also: script_visualize_orbits_templates_tsne.m 

clear; close all;

dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are stored
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/mds_tsne'; % dir to save figures
flag_print = true;

%% Load data
tagData = 'Pyritohedral_6';
filename_db = fullfile(dirData, sprintf('imdb_%s_sds_20', tagData));
load(filename_db);

% Data: training set
sd = 1; % seed

% Xr = X(:, ind_set==1, sd); ind_xr = ind_orbit(ind_set==1);
% N = sum(ind_t==1); % [d, Nx] = size(Xr); kOrbitSize = Nx./N;

%% Data: validation set
% Xt = X(:, ind_set==3, sd);
Xv_all = X(:, ind_set==2, sd); ind_xv_all = ind_orbit(ind_set==2);
Tv = T(:, ind_t==2, sd); 

%% Subsample validation set 
nOrbits = 50; % size(Xv_all, 2)
subScheme = 'decor';
s = rng; rng(0);
r = subSampleSet(Tv, nOrbits, subScheme);
rng(s);

Tvr = Tv(:,r);

%% Distance matrices
% figure; imagesc(project_unit_norm(Tvr)'*project_unit_norm(Tvr)); colorbar; %  cosine
% figure; imagesc(pdist2(Tvr', Tvr', 'euclidean')); colorbar; % euclidean

%% Subset of validation set used 
ind_orbit_val = r + min(ind_xv_all)-1;  
ind_points_from_orbit = ismember(ind_xv_all', ind_orbit_val', 'rows');
% sum(ind_points_from_orbit)

Xv = Xv_all(:, ind_points_from_orbit);
ind_xv = ind_xv_all(ind_points_from_orbit);

[d, Nv] = size(Xv); kOrbitSize = Nv./nOrbits;

clear X Xv_all ind_orbit filename_db sd_vec s dataGroupName r;


%% Visualization
index_t = 1:nOrbits; % template index
index_x = kron(index_t, ones(1, kOrbitSize)); % orbit index

dataVis = 'orbits'; 
tagDataVis = sprintf('%s_%s', tagData, dataVis);   
switch dataVis
    case 'templs'
     dataX = Tvr'; y = index_t; 
    case 'orbits'
     dataX = Xv'; y = index_x; 
end


no_dims = 2; 
typeVis = 'mds_regd'; 

switch typeVis
    
    case 'pca'
        
       %% Run PCA
        [~, ~, Xp] = eigpca(dataX);
        mappedXk = Xp(:, 1:no_dims); % projection on 2 main eigenvectors
        nTrials = 1;
        
    case 'mds_eucl'
        %% Classic MD scaling on Euclidean
        distMat = pdist2(dataX, dataX, 'euclidean'); 
        figure; imagesc(distMat); colorbar; % euclidean
        
        [mappedXk, ~] = cmdscale(distMat, no_dims);
        close;
        
    case 'mds_cos'
        %% Classic MD scaling on cosine distance
        distMat = pdist2(dataX, dataX, 'cosine'); 
        figure; imagesc(distMat); colorbar; % euclidean
        
        [mappedXk, ~] = cmdscale(distMat, no_dims);  
        close;
       
    case 'mds_regd'
        %% Classic MD scaling on regdist
         W = eye(d, kOrbitSize); % eye(d); %, kOrbitSize);
         s = 0.1; distMat = regWdistWrapper(W, dataX', s);
         figure; imagesc(distMat); colorbar; % euclidean         
         
         [mappedXk, ~] = cmdscale(distMat, no_dims);
         close;
        
    otherwise
        
        %% Run t-SNE
        perplexity = 30; nTrials = 10;
        mappedXk = mytsne(dataX, 'tsne', nTrials, no_dims, perplexity);

end

%% Visualization

mappedXk = zscore(mappedXk); % standardize scores in the 2 dimensions

fn = 14; markersize = 40;
h = figure; scatter(mappedXk(:,1), mappedXk(:,2), markersize, y, 'filled'); %, linewidth);
% myscatter(featOrbitMDS(:,1:no_dims), y, [],  arrayfun(@num2str, 1:nOrbits, 'Uniform', false), fn);
% axis equal; set(gca, 'box', 'off', 'zticklabel',[], 'xticklabel',[], 'yticklabel',[]); %, 'fontsize', fn); %, 'FontName', 'Helvetica')
axis tight; axis equal;
set(gca, 'box', 'off', 'fontsize', fn, 'FontName', 'Helvetica', 'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3])
% set(gca,'XTickLabel',num2str(get(gca,'XTick').'))
axis off;


%% Printing
% flag_print = false;
if flag_print
    
    % h1 = gcf;
    % set(h1, 'color', 'white'); set(h1, 'InvertHardCopy', 'off'); %pause(0.01)
    % set(h1, 'PaperPositionMode', 'manual'); set(h1,'Units','normalized','Position',  [0 0 0.5 5]);
    % set(h1, 'PaperPositionMode', 'auto');
    
    commonFileName = fullfile(dirFigs, sprintf('%s_%s_%d_d_N_%d', tagDataVis, typeVis, no_dims, nOrbits));
    printif(h, sprintf('%s', commonFileName), flag_print, false, false);
end

