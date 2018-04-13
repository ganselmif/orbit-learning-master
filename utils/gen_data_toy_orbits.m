% Generate data for different configurations and experiments
%
% This should be modified, used and called whenever any toy example is run!

function [X, k, d, filename_data, index_x, W] = gen_data_toy_orbits(typeData, nOrbits, typeInit)

if nargout==6 && nargin<3, typeInit = 'rand'; end
if nargin==0, typeData = 'group'; end

% rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random

%% Data
switch typeData
              
    case 'irotmnist'
                
        %% Directory (and name) of imdb structure
        % Warning: this is machine dependend. Need to change. 
        imdbPath = fullfile('/media/gevang/Data/work/code/matconvnet/data/irotmnist');   
        
        n_instance = 5; % number of instances per digit class
        rot_step = 10; % rotation step in degrees
        filename_data = fullfile(imdbPath, sprintf('imdb_n_%d_r_%d.mat', n_instance, rot_step));
        
        %% Load data
        load(filename_data);
        images.data = squeeze(images.data);
        
        %% Crop and subsample 
        subRatio = 3; % subsample ratio       
        images.data = images.data(2:subRatio:end-2,2:subRatio:end-2,:);
        
        [sa, sb, nImages] = size(images.data);         
        X = reshape(images.data, [sa*sb, nImages]);
        y = images.labels;
        
        % nOrbits = n_instance*length(unique(y));  % number of orbits
        % nOrbitSize = nImages/nOrbits; % samples per orbit
        
        %% Keep single "class" in MNIST 
        indClass = 4;
        X = double(X(:, y==indClass));
        nOrbits = n_instance;
        
        %% visualize orbits
        % figure;
        % display_network(X, false, true); % all instances/orbits from class c   
        
    case 'group'
        
        if nargin<2, nOrbits = 10; end % number of orbits
        
        groupNames = {'Cyclic', 'Dihedral', 'Dicyclic', 'Crystallographic'};
        groupDims = [6, 7, 15, 17, 25]; 
        
        groupTag = groupNames{1}; 
        groupDim = groupDims(1);      
        filename_data = sprintf('%sGroup%d.txt', groupTag, groupDim);
        
        % X = DatagenS(filename_group, N);
        X = genGroupData(filename_data, nOrbits);
        
    otherwise % assume group file name, e.g. '*Group*.txt' is given as input
        
        if isempty(strfind(typeData, '.txt'))
            filename_data = [typeData, '.txt'];
        else filename_data = typeData;
        end
        
        X = genGroupData(filename_data, nOrbits);
end

[d, nData] = size(X); % dimension of data
k = nData/nOrbits; % orbit size

%% Keep track of orbit membership/ID
index_x = kron(1:nOrbits, ones(1, k));

%% Shuffle 
r = randperm(nData);
X = X(:, r);
index_x = index_x(:,r); 

%% Add noise
% Gaussian noise
% X = X + randn(size(X));
% Drop columns
% nD = 20; X(:, randi(Nx, [1,nD])) = [];

%% Initilization vector
if nargout==6 % exist('W','var')
    switch typeInit
        %% Initialization
        case 'group'
            % W = DatagenS(filename_group,1);
            W = genGroupData(filename_data, 1);
            W = W + 0.5*randn([d, k]);
            
        otherwise
            % W = randn([d, k]);
            W = rand([d, k]);
            % W = double(squeeze(winit(d, 1, 1, k, 'initrand')));
    end
end

%% Normalize data X and initial weights W

% (to unit length)

% X = project_unit_norm(X);
% W = project_unit_norm(W);

% W = normsetvecs1(W);
% X = normsetvecs1(X);

