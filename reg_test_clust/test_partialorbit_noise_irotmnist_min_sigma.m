% Variation of test_partialorbit_noise_irotmnist.m to run for multiple
% sigma (i.e. minimizing the regularizer over multiple sigma values).
  
% Dependencies: regW_fixed_min_s.m

clear;

% experimental configuration params
dig = 1;
n = 0; a = 0; b = n; % noise params:
s = [eps 0.05 0.1 0.2 0.5 1];
pOrbSize = 5; % elements of orbit to be considered known
iter = 500;

%% rotated instance MNIST
% dirData = 'irotmnist';
dirData = '/media/gevang/Data/work/code/cbcl/orblearn/data';
dirFigs = '/media/gevang/Data/work/exp/orblearn/orbitrec/figs';

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
%figure; display_network(Xs, false, true, nOrbitSizeSub);
% Xs(Xs<0)=0;

%% normalize to unit length
C = project_unit_norm(Xs);

% select orbit as base
% dig = 1;
% pick elements as seeds
nPol = nOrbitSizeSub; % all

ind_dig = find(ys==dig);
ind_dig_s = ind_dig(1:nPol); % randi(nOrbitSizeSub, 1, 6));
ind_rest = setdiff(1:size(C, 2), ind_dig_s); % remaining elements to pick from

%% remaining elements
Cr = C(:, ind_rest);
length_rest = size(Cr, 2);
% Cr = Cr(:, randperm(length_rest));
% figure; display_network(Cr, false, true, nOrbitSizeSub);

%% true orbit
Oc = C(:, ind_dig_s); % true orbit
% figure; display_network(W1, false, true);
% [d, k] = size(W1);

%% Loop over different random false orbits and different re-shuffles of true orbit 
regd1 = nan(iter, nOrbitSizeSub); % null values 
regd2 = nan(iter, nOrbitSizeSub); % null values
regd3 = nan(iter, nOrbitSizeSub); % null values

for j=1:iter
    
    %% this IS the stochastic aspect of each iteration 
    W1 = Oc(:, randperm(nOrbitSizeSub)); % shuffle the elements of the true orbit also (each iter has a different order)
    W2 = Cr(:, randi(length_rest, nOrbitSizeSub)); % k random elements from the remaining set 
       
    % add iid noise
    if n~=0
        N = randArrayInRange([d, nOrbitSizeSub, 1], a, b);
        W1 = project_unit_norm(W1 + N);
        W2 = project_unit_norm(W2 + N);
    end
    
    for k = pOrbSize:nOrbitSizeSub
        E = kron(eye(k), ones(k));
        regd1(j, k) = regW_fixed_min_s(W1(:, 1:k), k, s, E);
        regd2(j, k) = regW_fixed_min_s(W2(:, 1:k), k, s, E); % all random orbit
        regd3(j, k) = regW_fixed_min_s([W1(:, 1:pOrbSize) W2(:, pOrbSize+1:k)], k, s, E); % first pOrbSize are fixed 
    end
    
    if 0
        figure; hold on;
        plot(regd1); plot(regd2);
        legend('orb','sorb'); hold off;
    end
end

% statistics across random trials
e1 = std(regd1, 0, 1);
e2 = std(regd2, 0, 1);
e3 = std(regd3, 0, 1);
m1 = mean(regd1, 1);
m2 = mean(regd2, 1);
m3 = mean(regd3, 1);

x = 1:nOrbitSizeSub;
figure; hold all; % plot(m1); hold;
errorbar(x, m1, e1, '.-'); %, 'linewidth', 2);
errorbar(x, m2, e2, '.-'); %, 'linewidth', 2);
errorbar(x, m3, e3, '.--'); %, 'linewidth', 2);
xlabel('# orbit elements' );
ylabel('regularizer value');
axis tight; grid on; ax = gca;
ax.XTick = 1:2:24;
title(sprintf('digit: %d, porb: %d, iter: %d', dig-1, pOrbSize, iter));
legend('true orbit (shuffled) (k)', 'random orbit (k)', 'random orbit (k-porb)')

printif(gcf, fullfile(dirFigs, sprintf('%d_partialorbit_irotmnist_porb_%d_iter_%d_min_sigma', dig, pOrbSize, iter)), true);

