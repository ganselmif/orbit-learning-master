
clear;
%% Comparison figures 
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs';


groupName = 'CyclicGroup13';
filename = [groupName '.txt'];


W = genGroupData(filename, 1);
[d, k] = size(W);

figure; imagesc(W);  axis equal; axis off; %colorbar;
printif(gcf, fullfile(dirFigs, [groupName '_weights']), true, false)

figure; imagesc(W'*W);  axis equal; axis off; %colorbar;
printif(gcf, fullfile(dirFigs, [groupName '_grammian']), true, false)

%% Noisy tests
% Test 1: AWGN noise  
W1 = W + 0.5*randn([d, k]);

% Test 2: noise
W2 = rand([d, k]);

% Test 3: 1 element 
W3 = W; W3(randi(d), randi(k)) = mean(W(:));

%% Regularizer values 
s = 0.001;
regW_fixed(W, k, s)
regW_fixed(W1, k, s)
regW_fixed(W2, k, s)
regW_fixed(W3, k, s)

% Change with noise 
rW1_all = [];
for i=1:200
    rW1 = [];
for sw = 0:0.01:1
    W1 = W + sw*randn([d, k]);
    rW1 = [rW1 regW_fixed(W1, k, s)];
end
rW1_all = [rW1_all; rW1];
end
figure; loglog((0:0.01:1), mean(rW1_all), 'linewidth', 2);
axis tight; grid on;
xlabel('noise variance'); ylabel('regularizer');
set(gca, 'fontsize', 12);
title('Adding noise (200 trials)', 'fontsize', 12)
printif(gcf, fullfile(dirFigs, [groupName '_noise']), true, true)


% Change with noise
rW3_all = [];
for i=1:200
rW3 = [];
for sw = 1:numel(W)
    W3 = W; W3(randi(numel(W),[sw,1])) = mean(W(:));
    rW3 = [rW3 regW_fixed(W3, k, s)];
end
rW3_all = [rW3_all; rW3];
end
figure; semilogy(1:numel(W), mean(rW3_all), 'linewidth', 2);
axis tight; grid on;
xlabel('number of changed elements'); ylabel('regularizer');
set(gca, 'fontsize', 12);
title('Changing n values in W (200 trials)', 'fontsize', 12)
printif(gcf, fullfile(dirFigs, [groupName '_noise_location']), true, true)


