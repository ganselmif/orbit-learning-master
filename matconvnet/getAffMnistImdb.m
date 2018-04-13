% Prepare the rotated MNIST imdb structure, returns image data with mean image subtracted
%
% See also: getMnistImdb.m getRotMnistImdb.m 

function imdb = getAffMnistImdb(opts, s)

if nargin==1, 
    s = 0.02; % number of sets to use 
end

% opts.dataDir = '/media/gevang/Data/data/affNIST/transformed';
sets = {'training_batches', 'validation_batches', 'test_batches'};

if ~exist(opts.dataDir, 'dir')
     mkdir(opts.dataDir);
 end
% 
% if ~exist(fullfile(opts.dataDir, files{1}), 'file')
%     url = sprintf('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip') ;
%     fprintf('downloading %s\n', url);
%     gunzip(url, opts.dataDir);
% end

%% train set
load(fullfile(opts.dataDir, sets{1}, '1.mat'));

N1 = s*numel(affNISTdata.original_id); 
indRand1 = randsplitho(double(affNISTdata.label_int'), N1);
N1 = length(indRand1);

y1 = []; x1 = [];
for b=1:32
    load(fullfile(opts.dataDir, sets{1}, sprintf('%d.mat',b)));
    y1(:, (b-1)*N1 + 1:N1*b) = affNISTdata.label_int(1, indRand1)+1;
    x1(:, (b-1)*N1 + 1:N1*b) = affNISTdata.image(:, indRand1); 
end

%% validation set
load(fullfile(opts.dataDir, sets{2}, '1.mat'));

N2 = s*numel(affNISTdata.original_id);
indRand2 = randsplitho(double(affNISTdata.label_int'), N2);
N2 = length(indRand2);

y2 = []; x2 = [];
for b=1:32
    load(fullfile(opts.dataDir, sets{1}, sprintf('%d.mat',b)));
    y2(:, (b-1)*N2 + 1:N2*b) = affNISTdata.label_int(1, indRand2)+1;
    x2(:, (b-1)*N2 + 1:N2*b) = affNISTdata.image(:, indRand2); 
end

%% test set
load(fullfile(opts.dataDir, sets{3}, '1.mat'));

N3 = s*numel(affNISTdata.original_id);
indRand3 = randsplitho(double(affNISTdata.label_int'), N3);
N3 = length(indRand3);

y3 = []; x3 = [];
for b=1:32
    load(fullfile(opts.dataDir, sets{3}, sprintf('%d.mat',b)));
    y3(:, (b-1)*N3 + 1:N3*b) = affNISTdata.label_int(1, indRand3)+1;
    x3(:, (b-1)*N3 + 1:N3*b) = affNISTdata.image(:, indRand3); 
end

set = [ones(1, numel(y1)) 2*ones(1, numel(y2)) 3*ones(1, numel(y3))];

% Remove mean estimated from train set
data = single(permute(reshape(cat(2, x1, x2), 40, 40, 1, []), [2 1 3 4]));
dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean);

data = single(permute(reshape(cat(2, x1, x2, x3), 40, 40, 1, []), [2 1 3 4]));

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = double(cat(2, y1, y2, y3));
imdb.images.set = set;

imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false);
% affNIST randomly sampled indices from each batch
imdb.meta.affNISTind.train = indRand1;
imdb.meta.affNISTind.val = indRand2;
imdb.meta.affNISTind.test = indRand3;


