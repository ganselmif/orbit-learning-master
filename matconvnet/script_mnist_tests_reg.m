% Dependencies:
% MatConvNet (fork: https://github.com/gevangel/matconvnet.git)
% OrbLearn

%%
% vl_setupnn;
addpath(vl_rootnn);
addpath(fullfile(vl_rootnn, 'examples', 'mnist')) ;
% addpath(fullfile(vl_rootnn(), 'examples', 'cifar')) ;
addpath(fullfile(vl_rootnn, 'examples', 'imagenet')) ;

%%
clear;
expDirRoot = vl_rootnn(); % '/media/gevang/Data/work/code/cbcl/matconvnet/';
resDirRoot = '/media/gevang/Data/work/exp/orblearn/matconvnet';
figsDirRoot = fullfile(resDirRoot, 'figs');

%% Dataset
tagTest = 'rotmnist';
% tagTest = 'mnist';
switch tagTest
    case 'mnist';
        dataDir = '/media/gevang/Data/data/MNIST/idx';
        imdbPath = fullfile(vl_rootnn, 'data/mnist', 'imdb.mat');
        
    case 'rotmnist'
        % rotated MNIST variation
        tagTest = 'rotmnist';
        dataDir = '/media/gevang/Data/data/MNIST_var/mnist_rotation_new';
        imdbPath = fullfile(vl_rootnn, 'data/rotmnist', 'imdb.mat');
        
    case 'irotmnist'
        % rotated MNIST variation
        tagTest = 'irotmnist';
        dataDir = '/media/gevang/Data/data/MNIST/idx';
        imdbPath = fullfile(vl_rootnn, 'data/irotmnist', 'imdb.mat');
        
    case 'affmnist'
        % affine MNIST variation
        tagTest = 'affmnist';
        dataDir = '/media/gevang/Data/data/affNIST/transformed';
        imdbPath = fullfile(vl_rootnn, 'data/affmnist', 'imdb.mat');    
end

%% Train
% Generic train options
trainOpts.gpus = 1;
trainOpts.continue = true;

%% Define experiment
networkType = {'simplenn'};
% modelType = {'cnn_2_layer'};
modelType = {'cnn_1_layer'};

ex.trainOpts = trainOpts;
ex.networkType = char(networkType);
ex.modelType = char(modelType);
ex.numEpochs = 100;
ex.batchSize = 100;
ex.useBatchNorm = false;

% regularization
ex.useReg = true;
ex.regType = 'dreg-m';
ex.regParam = 10;
if ex.useReg
    ex.useWeightNorm = true;
end
ex.useWeightNorm = false;

% expDir = fullfile(expDirRoot, ['data/test-mnist' suffix]);
% [net, info] = cnn_mnist(...
%     'expDir', expDir, ...
%     'dataDir', dataDir, ...
%     'useBatchNorm', false, ...
%     'networkType', ex.networkType, ...
%     'train', ex.trainOpts) ;

% naming conventions
prefix = ex.modelType;
suffix = ex.networkType;
nFilters = 25; % num filters: change this within cnn_mnist_init_mod.m
expName = sprintf('%s_w_%d_reg_%d', prefix, nFilters, ex.useReg);
if ex.useReg
    expName = sprintf('%s_%s_l_%d', expName, ex.regType, ex.regParam);
end
expDir = fullfile(resDirRoot, tagTest, expName);%, suffix));


% options for training
varargin_cnn =  {...
    'expDir', expDir, ...
    'dataDir', dataDir, ...
    'imdbPath', imdbPath, ...
    'batchNormalization', ex.useBatchNorm, ...
    'networkType', ex.networkType, ...
    'modelType', ex.modelType, ...
    'numEpochs', ex.numEpochs, ...
    'batchSize', ex.batchSize, ...
    'useWeightNorm', ex.useWeightNorm, ...
    'useReg', ex.useReg, ...
    'train', ex.trainOpts};

% regularization
if ex.useReg % isfield(opts, 'regType') && ~isempty(opts.regType)
    varargin_cnn = [varargin_cnn, {'regParam', ex.regParam, 'regType', ex.regType}];%, ..., ...
    %'regType', opts.regType};
end

% c1 = clock;
[net, info] = cnn_mnist_reg(varargin_cnn{:});
% c2 = clock;

% %% OR load the pre-trained CNN
% modelPath = fullfile(expDir, sprintf('net-epoch-%d.mat', ex.numEpochs));
% load(modelPath);

showVis = false;

flag_print = true;
if flag_print
    figsDir = fullfile(figsDirRoot, expName);
    if ~isdir(figsDir), mkdir(figsDir); end;
end

if showVis
           
    % display one layer network weights
    W1 = squeeze(net.layers{1}.weights{1});
    [size_wx, size_wy, filters_w] = size(W1);
    
    Wd = reshape(W1, size_wx*size_wy, filters_w);
    
    s = 0.0001;    
    
    if ~isfield(net.layers{1}, 'groups')
        %% Single orbit
        G = Wd'*Wd; % Gramian
        regW_fixed(Wd, filters_w, s)
        figure; display_network(Wd, false, true);
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_weights']), flag_print, false)
        figure; imagesc(Wd); axis equal; axis off;
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_weights_matrix']), false)
        figure; imagesc(G); axis equal; axis off; colorbar
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_weights_gramian']), flag_print, false)
        
    else
        %% Groups/Multiple orbits
        nGroups = length(unique(net.layers{1}.groups));
        groupSize =  net.layers{1}.groupSize;
        
        if strcmp(ex.regType, 'dreg-m')
            regW_mult(Wd, net.layers{1}.groups, groupSize, s)
        elseif strcmp(ex.regType, 'sreg')
            sregW_mult_cross(Wd, net.layers{1}.groups, groupSize)
        elseif strcmp(ex.regType, 'dreg-mc')
            regW_mult_cross(Wd, net.layers{1}.groups, groupSize, s)
        end
        
        % Visualize groups (each column is a group)
        figure; display_network(Wd, false, true, groupSize);
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_weights']), flag_print, false)
        
        figure; l = 0; G = [];
        for g=1:nGroups
            l = l + 1;
            subplot(2,3,g);
            Wdg = Wd(:,net.layers{1}.groups==g);
            G(:,:,l) = Wdg'*Wdg; % Gramian
            imagesc(G(:,:,l));  axis equal; axis off; %colorbar;
        end
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_weights_gramian']), flag_print, false)
        
        %% Cross-products
        %         l = 0; Gc = [];
        %         for g1=1:nGroups
        %             for g2=1:nGroups
        %                 Wdg1 = Wd(:,net.layers{1}.groups==g1);
        %                 Wdg2 = Wd(:,net.layers{1}.groups==g2);
        %                 l = l + 1;
        %                 reg(g1,g2) = regW_cross(Wdg1, Wdg2, groupSize, s);
        %                 Gc(:,:,l) = Wdg1'*Wdg2; % Gramian
        %             end
        %         end
        %         Mc = reshape(Gc, groupSize^2, l);
        %         figure; display_network(Mc, true, true);
        %
        %         figure;
        %         for i=1:l
        %             subplot(sqrt(l),sqrt(l),i);
        %             imagesc(Gc(:,:,i)); axis equal; axis off; % colorbar;
        %         end
        %         figure; imagesc(reg); colorbar; axis equal;
        
        
        
    end
    
    %     imdb = load(imdbPath);
    %     Xi = imdb.images.data(:,:,:,imdb.images.labels==2); % class 1
    %     Xi = squeeze(Xi); [sa, sb, sc] = size(Xi);
    %     % figure; vl_imarraysc(Xr, 'CMap', gray); % axis equal;
    %     Xi = reshape(Xi, sa*sb, sc);
    %     figure; display_network(Xi(:,1:100), false, true);
    
    
    %% display second layer network weights
    %     W2 = squeeze(net.layers{3}.weights{1});
    %     [size_wx, size_wy, filters_w, filter_dim] = size(W2);
    %     figure; display_network(reshape(W2, size_wx*size_wy, filters_w*filter_dim), false, true);
    
    print_train_cnn(info);
    printif(gcf, fullfile(figsDir, [tagTest '_' expName '_net_train']), flag_print, true)
    
    if ex.useReg
        print_train_reg_cnn(info, net.meta.trainOpts.regParam);
        printif(gcf, fullfile(figsDir, [tagTest '_' expName '_net_reg']), flag_print, true)
    end
    
end


% deploy/save trained network
net = cnn_imagenet_deploy(net); layers = net.layers; meta = net.meta;
netName = sprintf('%s_epoch_%d.mat', expName, ex.numEpochs);
% save(fullfile(expDir, sprintf('mnist-cnn.mat', ex.numEpochs)), 'layers', 'meta');
save(fullfile(expDir, netName), 'layers', 'meta');

%**************************************************************************

%% 1. load the pre-trained CNN
netName = sprintf('%s_epoch_%d.mat', expName, ex.numEpochs);
modelPath = fullfile(expDir, netName);
net = load(modelPath);

if 0
    %% MNIST data
    %imdbPath = fullfile(expDir, 'imdb.mat');
    imdb = load(imdbPath);
    
    % display some images (sanity check)
    [sa, sb, sc, sd] = size(imdb.images.data);
    nImages = 100; r = randi(sd, 1, nImages);
    
    % Xi = imdb.images.data(:,:,:,imdb.images.labels==2); % class 1
    % Xr = squeeze(Xi(:, :, :, r));
    Xr = squeeze(imdb.images.data(:, :, :, r));
    % figure; vl_imarraysc(Xr, 'CMap', gray); % axis equal;
    Xr = reshape(Xr, sa*sb, nImages);
    figure; display_network(Xr, false, true);
    
    printif(gcf, fullfile(figsDir, tagTest), true, false)
        
    %% Test set
    Xt = imdb.images.data(:,:,:,imdb.images.set==3); %imresize(imdb.images.data(:,:,:,imdb.images.set==3), net.meta.inputSize(1:2));
    Yt = imdb.images.labels(imdb.images.set==3);
    
    % run the CNN on Test Set (not for training is already included in val values)
    res = vl_simplenn(net, Xt);
    % y = vl_nnloss(res(end-1).x, Yt); %, [], 'type', 'softmaxlog');
    
    %% get classification result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    [maAc, perClass, miAc] = classaccuracy(Yt, best);
    maE = (1 - maAc)*100
    miE = (1 - miAc)*100
end


%**************************************************************************

%% 2. Using cnn_mnist_evaluate and the set field of imdb
%trainOpts.gpus = 1;
%trainOpts.continue = true;

ev.trainOpts.gpus = trainOpts.gpus;

res_info = cnn_mnist_evaluate(...
    'expDir', expDir, ...
    'modelPath', modelPath, ...
    'imdbPath', imdbPath, ...
    'train', ev.trainOpts);

fprintf('Test error: %0.4f\n', res_info.val.top1err(1))

