% Script testing the clustering properties of the regularizer function
clear;
% rng(0);

%% rotated instance MNIST
% dirData = 'irotmnist';
dirData = '/media/gevang/Data/work/code/cbcl/orblearn/data';

load(fullfile(dirData, 'imdb_sort_raw.mat'));
images.data = squeeze(images.data);
[sa, sb, nImages] = size(images.data);
d = sa*sb;

% matrix (d x K)
X = reshape(images.data, [d, nImages]);
% (K x 1) orbit label
y = images.labels;

nOrbits = length(unique(y)); % number of orbits
nOrbitSize = nImages/nOrbits; % samples per orbit

%% visualize orbits
% figure; for i=1:9, subplot(3,3,i); display_network(X(:, y==i), false, true); end
% figure;  display_network(X, false, true, nOrbitSize);

%% subsample regularily orbits
% C = reshape(squeeze(images.data), 28^2, 1200);%orbitsize=96
nStep = 5;
Xs = X(:, 1:nStep:nImages);
ys = y(:, 1:nStep:nImages);
nOrbitSizeSub = nOrbitSize/nStep; % sub-sampled orbit size
figure; display_network(Xs, false, true, nOrbitSizeSub);
% Xs(Xs<0)=0;

%% normalize to unit length
C = project_unit_norm(Xs);
figure; display_network(C, false, true, nOrbitSizeSub);

%% ************************************************************************

% select orbit as base
dig = 9;
ind_dig = find(ys==dig);
% pick "pol" elements as seeds
nPol = 6;
ind_dig_s = ind_dig(1:nPol); % randi(24, 1, 6));
ind_rest = setdiff(1:size(C, 2), ind_dig_s); % remaining elements to pick from

Oc = C(:, ind_dig_s); % starting orbit

%% shuffle the remaining elements
Cr = C(:, ind_rest);
length_rest = size(Cr, 2);
Cr = Cr(:, randperm(length_rest));
% figure; display_network(Cr, false, true, nOrbitSizeSub);

s = 1;
for j=1:nOrbitSizeSub-nPol
    % loop over remaining slots in the orbit
    
    % calculate value of regularizer for a subset from every other element
    reg = [];
    k =  size(Oc, 2) + 1;
    E = kron(eye(k), ones(k));
    
    for i = 1:length_rest;
        % loop over remaining elements
        Ot = [Oc, Cr(:,i)];
        reg(i) = regW_fixed(Ot, k, s, E);
    end
    % get minimizer of 'augemented orbit'
    [~, min_ind] = min(reg);
    if length(min_ind)>1
        print('conflict... same value of minimum in more than one element');
    end
    
    Oc(:, end+1) = Cr(:,min_ind); % augment orbit
    Cr(:,min_ind) = []; % remove chosen element
    length_rest = length_rest - 1;
end


%% display chosen orbit
figure; display_network(Oc, false, true); title('retreived orbit');
figure; display_network(C(:, ind_dig), false, true); title('ground truth orbit');

figure;
subplot(1,2,1); imagesc(Oc'*Oc); title('retreived orbit'); % resulting orbit
subplot(1,2,2); imagesc(C(:, ind_dig)'*C(:, ind_dig)); title('ground truth orbit'); % ground_truth

