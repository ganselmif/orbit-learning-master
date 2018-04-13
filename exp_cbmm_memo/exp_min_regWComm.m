% Experiments for minimizing reg(W) and comm(WWt, XXt)
%
% Taken from: script_min_regWComm.m

clear; close all;
debug = false; % flag for debugging

%% Parameters
s = 0.1; % spread of Gaussian approximating the Dirac
lambda_w = 10^2; % regcomm weight

%% Data generation
rng(42, 'twister'); % Fix seed for reproducible results, comment if you want them random
% typeData = 'irotmnist'; [X, k, d, filename_data, W] = gen_data_toy_orbits(typeData);
typeData = 'DihedralGroup6'; N = 10; % [X, k, d, filename_data] = gen_data_toy_orbits(typeData, N);
filename_data = [typeData '.txt']; X = genGroupData(filename_data, N); [d, Nx] = size(X); k = Nx./N;
% X = X(:, randi1:end);
% X = X(:,randi(size(X,2), [1,1000]));

%% Auxiliary constants for gradient/regularizer
E = kron(eye(k), ones(k));
Ik = sparse(eye(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
[C, R] = gradW_opt_aux(k);
CRt = R'*C';

%% Cached computations
% XW = X'*W;
XXt = X*X'/Nx; % (Nx-1);
% XXt = normdotprod(X, X);

%% ************************************************************************
%% Minimization with Quasi-Newton and line-search: See also: script_min_regW.m

%% Compact constraint
costFunc = @(t)regWCommFuncGradVec(t, XXt, k, d, s, lambda_w);
%% Compact + all columns contraints
% costFunc = @(t)regWCommColumnsFuncGradVec(t, XXt, k, d, s, lambda_w);
%% Compact constraint + Log(Det(Cov)))
% lambda_c = 0.001;
% costFunc = @(t)regWCommDetCovFuncGradVec(t, XXt, k, d, s, lambda_w, lambda_c);

%%  Initialization
% rng('shuffle', 'twister'); scurr = rng;
rng(0, 'twister'); scurr = rng;
W = randSampleVec(d, k, 'uball');
% vecW = vec(W);

%% Optimization
% optimType = 'fminuncms'; %'fminunc';
ITER = 5000;

% case {'fminuncms'}
% Unconstraint/Multiple start points
tol = 10^-8;
numStartPoints = 10;

if isempty(gcp), parpool; end
[vecW, multiMinStruct] = fminMsWrapper(costFunc, vec(W), numStartPoints, 'uball', true, ITER, 'off', tol);
delete(gcp);

% Retrieve all points and solutions ordered by min f val
for c = 1:numStartPoints
    matVecW(:, c) = multiMinStruct(c).X; % converged points/solutions
    matVecWo(:, c) = multiMinStruct(c).X0{1}; % initial random points
    fval = [multiMinStruct(:).Fval]; % min fval
end



%% Metrics/Regularizers
We = reshape(matVecW, [d, k, numStartPoints]);
W = reshape(matVecWo, [d, k, numStartPoints]);

% We = reshape(vecW, d, k);

for c=1:numStartPoints
    reg_Wo(c) = regW_fixed(W(:,:,c), k, s, kE_term);     % original W
    reg_W(c) = regW_fixed(We(:,:,c), k, s, kE_term);     % converged W
    reg_XW(c) = regW_fixed(X'*We(:,:,c), k, s, kE_term); % X'W
    com_Wo(c) = norm(comm(XXt, W(:,:,c)*W(:,:,c)'), 'fro'); % commutator norm
    com_W(c) = norm(comm(XXt, We(:,:,c)*We(:,:,c)'), 'fro'); % commutator norm
    % det_a = log(det(We(:,:,c)*We(:,:,c)'))/d;
    % rege_Wo = []; % regE(W, XXt, d, k, s); % Eigenvalue reg (original W)
    % rege_W = []; % regE(We, XXt, d, k, s); % Eigenvalue reg (converged W)
    
    % fprintf('c: %2d, reg(Wo): %e, reg(W): %e, regE(Wo): %e, regE(W): %e, reg(X''W): %e, norm_com(Wo): %e, norm_com(W): %e, det_cov: %f\n', ...
    %    c, reg_Wo, reg_W, rege_Wo, rege_W, reg_XW, com_Wo, com_W, det_a)
    fprintf('c: %2d, reg(Wo): %e, reg(W): %e, reg(X''W): %e, norm_com(Wo): %e, norm_com(W): %e\n', ...
        c, reg_Wo(c), reg_W(c), reg_XW(c), com_Wo(c), com_W(c))
end


%% Invariance Test/Comparison signatures
nTrials = 500; % random templates to test

[ds_in, ds_out, Wo] = ComparisonGD(filename_data, We, false, nTrials); %,'CyclicGroup6.txt');
nBins = 50;
[stats_ds_in, stats_ds_out] = dispComparisonGD(ds_in, ds_out, nBins);


%% PRINT PLOTS
flag_print = false;

if flag_print
    
    % file naming    
    tagGroup = 'DihedralGroup6';
    figsDir = '/media/gevang/Data/work/exp/orblearn/figs';
    commonFileName = fullfile(figsDir, sprintf('%s_l_%d_N_%d_nS_%d', tagGroup, lambda_w, N, numStartPoints));
    
    
    h = figure(1); grid off;
    set(h.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], ...
        'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    % set(gcf, 'title', 'off');
    set(findall(h, '-property', 'FontSize'), 'FontSize', 14)
    
    printif(h, sprintf('%s_%s', commonFileName, 'within'), flag_print, true, true);
    
    h = figure(2); grid off;
    set(h.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], ...
        'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
        'LineWidth', 2);
    % set(gcf, 'title', 'off');
    set(findall(h, '-property', 'FontSize'), 'FontSize', 14)
    
    printif(h, sprintf('%s_%s', commonFileName, 'across'), flag_print, true, true);
    
    % figure; hold all;
    % bcomb = [fval./fval(1); reg_W./reg_W(1); reg_XW./reg_XW(1); com_W./com_W(1)]';
    % hb = bar((1:numStartPoints)', bcomb); colormap('default');
    % legend('fval', 'reg_W', 'reg_XW', 'com_W'); axis tight; grid minor;
    
    figure; hold all;
    bcomb = [(fval-mean(fval))./std(fval); (reg_W-mean(reg_W))./std(reg_W); (reg_XW - mean(reg_XW))./std(reg_XW); (com_W - mean(com_W))./std(com_W)]';
    % hb = plot(bcomb, 'linewidth', 2);
    hb = bar(bcomb, 'hist');
    % set(hb, 'LineStyle', 'none', 'Marker', 'o')
    % stem(bcomb, '.','linewidth', 2);
    legend('fval', 'reg(W)', 'reg(XW)', 'com(X,W)', 'location', 'best'); axis tight; % grid minor;
    xlabel('point index (ordered by fval)')
    ylabel('z-normalized value')
    
    h = gcf;
    set(h.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], ...
        'YMinorTick', 'on', 'YGrid', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'XTick', 1:1:20, 'XTickLabel','',...
        'LineWidth', 1)
    set(findall(h, '-property', 'FontSize'), 'FontSize', 14)
    
    printif(h, sprintf('%s_%s', commonFileName, 'metrics'), flag_print, true, true);
end
