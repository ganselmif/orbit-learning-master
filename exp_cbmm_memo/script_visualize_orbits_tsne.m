% Script to make tSNE/PCA visualizations of orbits and orbit training sets 

% GE, CBMM/LCSL/MIT, gevang@mit.edu

clear; % close all; clc;

%% Data generation
rng(0, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, k, d, filename_data, W] = gen_data_toy_orbits(typeData);

typeData = 'DihedralGroup6'; N = 100; % [X, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
filename_data = [typeData '.txt']; 

typeTempls = 'mball';
switch typeTempls
    case 'uball'
        [X, T] = genGroupData(filename_data, N);
    case 'mball'
        M = 20; % number of balls to subsample from
        [X, T] = genGroupData(filename_data, N, M);
end
[d, Nx] = size(X); k = Nx./N;

index_t = 1:N; % template index
index_x = kron(index_t, ones(1, k)); % orbit index


%% tSNE 
% dataX = T'; y = index_t; tagData = sprintf('%s_templs', typeTempls);
dataX = X'; y = index_x; tagData = sprintf('%s_orbits', typeTempls);

no_dims = 2; perplexity = 10; nTrials = 6;
typeVis = 'tsne'; %tSNE'

switch typeVis
    
    case 'pca'
        
       %% Run PCA
        [~, ~, Xp] = eigpca(dataX);
        mappedXk = Xp(:, 1:no_dims); % projection on 2 main eigenvectors
        nTrials = 1;
        
    case 'mds'
        %% Classic MD scaling
        distMat = pdist2(dataX, dataX, 'euclidean'); 
        [mappedXk, ~] = cmdscale(distMat, no_dims);
        
    otherwise
        
        %% Run t-SNE
        mappedXk = mytsne(dataX, 'tsne', nTrials, no_dims, perplexity);

end

%% Visualization
classInd = y; 
% classLab = unique(y);
fn = 14;
markersize = 40; 

h = figure; 
scatter(mappedXk(:,1), mappedXk(:,2), markersize, y, 'filled'); %, linewidth);
% gscatter(mappedXk(:,1), mappedXk(:,2), y, [] ,'o', markersize, 'doleg', 'off'); %, linewidth);
% myscatter(mappedXk(:,1:no_dims), y, classInd, [], fn); 
axis equal; 
set(gca, 'box', 'off', 'zticklabel',[], 'xticklabel',[], 'yticklabel',[], 'fontsize', fn, 'FontName', 'Arial')    

%% Printing
flag_print = false;
if flag_print
    
    % h1 = gcf;
    % set(h1, 'color', 'white'); set(h1, 'InvertHardCopy', 'off'); %pause(0.01)
    % set(h1, 'PaperPositionMode', 'manual'); set(h1,'Units','normalized','Position',  [0 0 0.5 5]);
    % set(h1, 'PaperPositionMode', 'auto');
    
    tagGroup = 'DihedralGroup6';
    figsDir = '/media/gevang/Data/work/exp/orblearn/figs';
    commonFileName = fullfile(figsDir, sprintf('%s_%s_N_%d_d_%d_nTrials_%d', typeVis, tagGroup, size(X,1), no_dims, nTrials));
    
    printif(h, sprintf('%s_%s', commonFileName, tagData), flag_print, true, true);
end