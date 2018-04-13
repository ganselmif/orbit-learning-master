% Generate irotMNIST data: single instance, all rotations per r degrees

%% Directory with original MNIST data 
dataDir = '/media/gevang/Data/data/MNIST/idx'; 

opts.dataDir = dataDir;
% Download, unzip MNIST files (if not there)
files = {'train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte', ...
    't10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte'} ;
if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir) ;
end
for i=1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, opts.dataDir) ;
    end
end

%% Directory (and name) of imdb structure
imdbPath = fullfile(vl_rootnn, 'data/irotmnist');
if ~exist(imdbPath, 'dir')
    mkdir(imdbPath);
end

%% Generate data 
n_instance = 5; % number of instances per digit class
rot_step = 10; % rotation step in degrees 
make_val = false;
rem_mean = false;
imdb = getMnistSingleRotInstImdb(opts, n_instance, rot_step, make_val, rem_mean); % structure that holds data

% save data in this directory 
opts.imdbPath = fullfile(imdbPath, sprintf('imdb_n_%d_r_%d.mat', n_instance, rot_step));
save(opts.imdbPath, '-struct', 'imdb') ;


%% EXAMPLE: Load data 
load(fullfile(opts.imdbPath));
images.data = squeeze(images.data);
[sa, sb, nImages] = size(images.data);
d = sa*sb;

% matrix (d x K)
X = reshape(images.data, [d, nImages]);
% (K x 1) orbit label
y = images.labels;

nOrbits = n_instance*length(unique(y));  % number of orbits
nOrbitSize = nImages/nOrbits; % samples per orbit


%% EXAMPLE: visualize orbits
figure; display_network(X(:, 1:n_instance*nOrbitSize), false, true, nOrbitSize); % first few orbits 
figure; c = 4; display_network(X(:, y==c), false, true, nOrbitSize); % all instances/orbits from class c
% figure; display_network(X, false, true, nOrbitSize); % everything 
figure; display_network(X(:, randperm(nImages, 100)), false, true); % 100 random elements


%% UNUSED: Order train/validation so that consequtive samples correspond to the same
if 0
    % imdb = load(imdbPath);
    [~, ind_sort] = sort(imdb.images.labels);
    imdb.images.data = imdb.images.data(:,:,:,ind_sort);
    imdb.images.labels = imdb.images.labels(ind_sort);
    imdb.images.set = imdb.images.set(ind_sort);
    imdbPath = fullfile(vl_rootnn, 'data/irotmnist', 'imdb_sort_raw.mat');
    save(imdbPath, '-struct', 'imdb') ;
end