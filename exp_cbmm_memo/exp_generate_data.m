% Generate random orbit datasets (train, test, val) for CBMM memo
%
% %% Example: load pre-computed data from drive
% dirExp = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located
%
% dim_group = 6; % group size
% tagData = 'Cyclic';
% filename_db = sprintf('imdb_%s_%d_sds_%d.mat', tagData, dim_group, 20);
% load(fullfile(dirExp, filename_db));

% GE, CBMM/LCSL/MIT, gevang@mit.edu

clear;

dirDataTxt = '/media/gevang/Data/work/code/orblearn/datasets_groups/group_matrices';
dirExp = '/media/gevang/Data/work/exp/orblearn/data'; % dir to save mat files

%% Group names
dim_group = 6;
dirData = sprintf('%s/dim%d', dirDataTxt, dim_group);
switch dim_group
    case 12
        dataGroupName = {'Cyclic', 'Dicyclic'};
    case 6
        dataGroupName = {'Cyclic', 'Pyritohedral', 'Dihedral'};
    case 4
        dataGroupName = {'Alternating', 'Cyclic', 'Dihedral', 'Symmetric', 'Tetrahedral', 'Vierergruppe'};
end

%% Sizes of generated sets
nTrain = 1000; % training
nVal = 0.2*nTrain; % validation
nTest = 500; % test

nOrbits = nTrain + nTest + nVal;

% index: train/val/test set
ind_t = [ones(1, nTrain) 2*ones(1, nVal) 3*ones(1, nTest)];

%% Seeds and sampling
sd_vec = 1:20;
samplType = 'uball'; % random vector sampling type

for g=1:length(dataGroupName)
    
    tagData = dataGroupName{g};
    filename_db = sprintf('imdb_%s_%d_sds_%d.mat', tagData, dim_group, length(sd_vec));
    filename_data = fullfile(dirData, sprintf('%s%d.txt', tagData, dim_group));
    clear X T k ind_x
    
    for sd = 1:length(sd_vec);
        
        rng(sd_vec(sd), 'twister'); s(sd) = rng;
        
        % templates
        T(:,:,sd) = randSampleVec(dim_group, nOrbits, samplType);
        
        % orbit data
        X(:,:,sd) = genGroupData(filename_data, [], T(:,:,sd));
        
    end
    
    [~, nData] = size(X(:,:,sd)); % dimension of data
    k = nData/nOrbits; % orbit size
    
    % index: orbit membership/ID
    ind_orbit = kron(1:nOrbits, ones(1, k));
    
    % index: set membership
    ind_set = [ones(1, nTrain*k) 2*ones(1, nVal*k) 3*ones(1, nTest*k)];
    
    save(fullfile(dirExp, filename_db), 'X', 'T', 'ind_orbit', 'ind_set', 'ind_t', 'sd_vec', 's');
    
end


