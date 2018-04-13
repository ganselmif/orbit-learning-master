% Script to minimize reg(W) with cross products
%
% See also: script_min_regWComm.m script_min_regWX.m

%% Cross-products 

clear;
rng(0); % 'default');

filename = 'DihedralGroup6.txt';
% W = DatagenS(filename,1);
M = 10;
W = genGroupData(filename, M);
[d, K] = size(W);

k = K/M; 
setm = kron(1:M, ones(1, k));

%% add noise 
W1 = W + 0.5*randn([d, K]);
Wn = W1; 

% normalize 
Wn = bsxfun(@minus, Wn , mean(Wn));
Wn = bsxfun(@rdivide, Wn, sqrt(sum(Wn.^2)));

lambda1 = 1; lambda2 = 1; s1 = 0.01; s2 = 1;
costFunc1 = @(t)(regWCrossFuncGradVec(t, setm, k, lambda1, 0, s1, s2));
costFunc2 = @(t)(regWCrossFuncGradVec(t, setm, k, lambda1, lambda2, s1, s2));

optimType = 'fminunc';% 'minFunc';

vecWe = fminWrapper(costFunc1, Wn(:), optimType, 1000, 'on');
We1 = reshape(vecWe, d, K);

figure; imagesc(We1'*We1); colorbar;
J1 = costFunc1(vecWe)


nGroups = M; % length(unique(setm));
figure; l = 0; G = [];
for g=1:nGroups
    l = l + 1;
    subplot(2,5,g);
    Wdg = We1(:,setm==g);
    reg(g) = regW(Wdg, k, 0.01); 
    G(:,:,l) = Wdg'*Wdg; % Gramian
    imagesc(G(:,:,l));  axis equal; axis off; colorbar;
    title(reg(g));
end
% printif(gcf, fullfile(figsDir, [expName '_weights_gramian']), true)

% Cross-products
l = 0; Gc = [];
for g1=1:nGroups
    for g2=1:nGroups
        Wdg1 = We1(:,setm==g1);
        Wdg2 = We1(:,setm==g2);
        l = l + 1;
        % reg(g1,g2) = regW_cross(Wdg1, Wdg2, 8, 0.01);
        Gc(:,:,l) = Wdg1'*Wdg2; % Gramian
    end
end
Mc = reshape(Gc, k^2, l);
figure; display_network(Mc, true, true);


vecWe = fminWrapper(costFunc2, Wn(:), optimType, 1000, 'on');
We1 = reshape(vecWe, d, K);

figure; imagesc(We1'*We1); colorbar;
J1 = costFunc2(vecWe)


