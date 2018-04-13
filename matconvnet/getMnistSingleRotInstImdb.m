% Prepare one instance, rotated MNIST imdb structure, returns image data with mean image subtracted
%
% See also: getMnistImdb.m

function imdb = getMnistSingleRotInstImdb(opts, n_instances, rot_step, make_val, rem_mean)

if nargin<5, rem_mean = false; end % flag to remove mean, default is true 
if nargin<4, make_val = true; end % make 20% validation set from generated data
if nargin<3, rot_step = 3; end % rotation step
if nargin<2, n_instances = 1; end % number of instances to rotate

% files = {'train-images-idx3-ubyte', ...
%     'train-labels-idx1-ubyte', ...
%     't10k-images-idx3-ubyte', ...
%     't10k-labels-idx1-ubyte'} ;
%
% if ~exist(opts.dataDir, 'dir')
%     mkdir(opts.dataDir) ;
% end
%
% for i=1:4
%     if ~exist(fullfile(opts.dataDir, files{i}), 'file')
%         url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
%         fprintf('downloading %s\n', url) ;
%         gunzip(url, opts.dataDir) ;
%     end
% end

f = fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1 = fread(f,inf,'uint8');
fclose(f);
x1 = permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

% f = fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
% x2 = fread(f,inf,'uint8');
% fclose(f);
% x2 = permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f = fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1 = fread(f,inf,'uint8');
fclose(f);
y1 = double(y1(9:end)')+1;

% f = fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
% y2 = fread(f,inf,'uint8');
% fclose(f);
% y2 = double(y2(9:end)')+1;


%%
lab = unique(y1);
Xc = [];
Yc = [];
for yc=1:length(lab)    
    xc = x1(:,:, y1==yc);
    sel = randperm(size(xc, 3), n_instances);
    % slower but more convenient: store rotations and then multiple instances
    for n=1:n_instances
        xc_rot = sampleRotateMNIST(reshape(xc(:,:,sel(n)), 1, 28*28), 1, rot_step);
        sc = size(xc_rot, 1);
        Xc = cat(3, Xc, reshape(xc_rot', 28, 28, sc));
        Yc = [Yc yc*ones(1, sc)];
    end    
end

% X = reshape(Xc, 28^2, 1200); figure; displayData(X(:,1:120)');

%% Validation set split
if make_val
    % Validation set (0.02 of generated data)
    indVal = randsplitho(Yc', 0.2)';
    indTrain = setdiff(1:length(Yc), indVal);
    
    x1 = Xc(:, :, indTrain); y1 = Yc(indTrain);
    x2 = Xc(:, :, indVal); y2 = Yc(indVal);    
    
    set = [ones(1, numel(y1)) 2*ones(1, numel(y2))];
    data = single(reshape(cat(3, x1, x2), 28, 28, 1,[]));
    labels = cat(2, y1, y2);
else
    % No validation set split
    data = single(reshape(Xc, 28, 28, 1, []));
    set = ones(1, numel(Yc));
    labels = Yc;
end

%% Remove mean estimated from train set
dataMean = mean(data(:,:,:,set == 1), 4);
if rem_mean
    data = bsxfun(@minus, data, dataMean);
end

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = labels;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);