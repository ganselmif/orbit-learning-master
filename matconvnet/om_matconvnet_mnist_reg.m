% Dependencies:
% MatConvNet (gevangel fork)
% OrbLearn

function om_matconvnet_mnist_reg(numEpochs, regParam, numGpus, tagSet)

if nargin<4, tagSet = 'rotmnist'; end
if nargin<3, numGpus = 1; end
if nargin<2, regParam = 0; end
if nargin<1, numEpochs = 100; end
if regParam
    useReg = true;
else
    useReg = false;
end

%%
addpath(vl_rootnn);
addpath(fullfile(vl_rootnn, 'examples', 'mnist')) ;
addpath(fullfile(vl_rootnn, 'examples', 'imagenet')) ;

% expDirRoot = vl_rootnn(); % '/media/gevang/Data/work/code/cbcl/matconvnet/';
resDirRoot = '/home/gevang/exp/orblearn/matconvnet/';
dataDirRoot = '/om/user/gevang/data/';

%% Dataset
switch tagSet
    case 'mnist';
        dataDir = fullfile(dataDirRoot, 'MNIST/idx');
        imdbPath = fullfile(dataDirRoot, 'matconvnet/mnist', 'imdb.mat');
    case 'rotmnist'
        % rotated MNIST variation
        tagSet = 'rotmnist';
        dataDir = fullfile(dataDirRoot, 'MNIST_var/mnist_rotation_new');
        imdbPath = fullfile(dataDirRoot, 'matconvnet/rotmnist', 'imdb.mat');
end

%% Train
% Generic train options
trainOpts.gpus = numGpus;
trainOpts.continue = true;

%% Define experiment
networkType = {'simplenn'};
modelType = {'dnn_1_layer'};

ex.trainOpts = trainOpts;
ex.networkType = char(networkType);
ex.modelType = char(modelType);
ex.numEpochs = numEpochs;
ex.useReg = useReg;
ex.regParam = regParam;

%% Naming 
prefix = ex.modelType;
% suffix = ex.networkType;
nFilters = 25; % num filters: change this within cnn_mnist_init_mod.m
expName = sprintf('%s_w_%d_reg_%d', prefix, nFilters, ex.useReg);
if ex.useReg
    expName = sprintf('%s_l_%d', expName, ex.regParam);
end
expDir = fullfile(resDirRoot, tagSet, expName);%, suffix));


%% TRAINING: options 
varargin_cnn =  {...
    'expDir', expDir, ...
    'dataDir', dataDir, ...
    'imdbPath', imdbPath, ...
    'useBatchNorm', false, ...
    'networkType', ex.networkType, ...
    'modelType', ex.modelType, ...
    'numEpochs', ex.numEpochs, ...
    'useReg', ex.useReg, ...
    'train', ex.trainOpts};

% regularization
if useReg % isfield(opts, 'regType') && ~isempty(opts.regType)
    varargin_cnn = {varargin_cnn{:}, ...
        'regParam', ex.regParam}; % ..., ...
        %'regType', opts.regType};
end

% c1 = clock;
[net, info] = cnn_mnist_reg(varargin_cnn{:});
% c2 = clock;

% %% OR load the pre-trained CNN
% modelPath = fullfile(expDir, sprintf('net-epoch-%d.mat', ex.numEpochs));
% load(modelPath);

if 0
    % deploy/save trained network
    net = cnn_imagenet_deploy(net); layers = net.layers; meta = net.meta;
    netName = sprintf('%s_epoch_%d.mat', expName, ex.numEpochs);
    % save(fullfile(expDir, sprintf('mnist-cnn.mat', ex.numEpochs)), 'layers', 'meta');
    save(fullfile(expDir, netName), 'layers', 'meta');
end