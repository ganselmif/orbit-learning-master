% Generating single-orbit examples for known groups
%
% Figures: CBMM Memo

clear;
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/group_examples';
flag_print = false;

%% Generate Data

dim_group = 6; %4

switch dim_group
    case 6
        dirData = sprintf('/media/gevang/Data/work/code/orblearn/datasets_groups/group_matrices/dim%d', dim_group);
        dataGroupName = {'Cyclic', 'Pyritohedral', 'Dihedral'};
    case 4
        dirData = sprintf('/media/gevang/Data/work/code/orblearn/datasets_groups/group_matrices/dim%d', dim_group);
        dataGroupName = {'Alternating', 'Cyclic', 'Dihedral', 'Tetrahedral', 'Vierergruppe'};
end

%% Single template
n =1;

samplType = 'usimp'; % random vector sampling type
% templates
t = (1:dim_group)'; t = t./norm(t);

cmap_w = lines(dim_group);
cmap_g = 'default';
fn = 13;

for g=1:length(dataGroupName)
    tagData = dataGroupName{g};
    
    filename_data = fullfile(dirData, sprintf('%s%d.txt', tagData, dim_group));
    
    W = genGroupData(filename_data, [], t);
    
    % Gramian
    G = W'*W;
    
    commonFileName = sprintf('%s%d', tagData, dim_group);
    
    % h = figure; imagesc(W); axis equal; axis off; colormap(cmap_w);
    % printif(h, fullfile(dirFigs, sprintf('%s_%s', commonFileName, 'W')), flag_print, false, false);
    h = imagescpix(W, cmap_w, fn);  axis equal; axis off;
    %printif(h, fullfile(dirFigs, sprintf('%s_%s_nums', commonFileName, 'W')), flag_print, false, false);
    
    
    %h = figure; imagesc(G); axis equal; axis off; colormap(cmap_g);
    %printif(h, fullfile(dirFigs, sprintf('%s_%s', commonFileName, 'G')), flag_print, false, false);
    h = imagescpix(G, cmap_g, fn);  axis equal; axis off;
    % printif(h, fullfile(dirFigs, sprintf('%s_%s_nums', commonFileName, 'G')), flag_print, false, false);
    pause; close all;
end

