% Frame potential plots and matrices ordered by signature std
%
% run:
% 1. om_min_regWCommReLU_cv

% see also: plot_matrices.m

flag_print = false;
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs';
dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located

% dataGroupName = {'Cyclic_6', 'Pyritohedral_6', 'Dihedral_6'}; 
tagData = 'Dihedral_6'; %dataGroupName{3};
tagLoss = 'regWCommReLU_300'; sd = 1; numStartPoints = 50; nLambdas = 8;
% commonFileName =  sprintf('%s_%s_startpoints_%d', tagLoss, tagData, numStartPoints);
commonFileName = sprintf('%s_%s_lw_%d_startpoints_%d', tagLoss, tagData, nLambdas, numStartPoints);
   
switch tagData
    case 'Cyclic_6';
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Cyclic_6_sds_20');
        n_sol = 50;
        
    case 'Dihedral_6'
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Dihedral_6_sds_20');
        n_sol = 25;
        
    case 'Pyritohedral_6';
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Pyritohedral_6_sds_20');
        n_sol = 15;
end
load(filename_cv);
load(filename_db);


%% Commutator lamba
% lambda_w_vals = 10.^(0:1:4);
% nLambdas = length(lambda_w_vals);

%% Define sets

% validation and test set (same seed)
Xv = X(:, ind_set==2, sd); ind_xv = ind_orbit(ind_set==2);
% Xt = X(:, ind_set==3, sd);

clear X ind_orbit filename_db sd_vec s dataGroupName r; 

%% ************************************************************************
%% Signature evaluation on validation set
% nDics = numStartPoints;
poolMethodNames = {'max', 'l2', 'lp', 'meanrelu', 'rms', 'mean', 'hist', 'moments', 'centmom', 'sumstats', 'cdf', 'histc'};
distMethodNames = {'euclidean', 'cosine', 'seuclidean', 'chebychev', 'minkowski'};

% Pool and distance method
poolMethod = poolMethodNames{2};
distMethod = distMethodNames{1};
typeSignature = 'single';

%% Validation/trials set
[d, kOrbitSize, nStartPoints, nLambdas, nLambdas_s] = size(We_lambda);
nComp = 1;
Nv = sum(ind_t==2);

% util matrices
[ind_in, ind_out] = indMatInterIntraOrbit(Nv, kOrbitSize);
index_point = kron(1:numStartPoints, ones(1, kOrbitSize))'; % orbit index: for pooling
index_pool = kron(1:nComp, ones(1, kOrbitSize))'; % orbit index: for pooling

clear dist_in dist_out frame_pot_W

% %% Self-Coherence lambda
% lambda_s_vals = 10.^(-3:1:0);
% nLambdas_s = length(lambda_s_vals);

%for ls = 1:nLambdas_s
% Self-coherence constant
s = 1;
% lambda_s = lambda_s_vals(s);
for l = 1:nLambdas
    
    D = reshape(We_lambda(:,:,:,l,s), [d, kOrbitSize*numStartPoints]);
    
    % Single component signature
    for indComp=1:nStartPoints
        We = D(:, index_point==indComp);
        
        % frame potential: sum of off-diagonal elements
        frame_pot_W(l,indComp) = sum(sum((normdotprod(We, We) - eye(kOrbitSize)).^2));
        % distances
        [dist_in(:,l,indComp), ~] = makeSignatureDist(Xv, We, poolMethod, distMethod, index_pool, ind_in, ind_out);
    end
end
%end

std_dist_in_W = squeeze(std(dist_in));
% std_dist_out_W = squeeze(std(dist_out));

% vectorize and sort
std_dist_in_W_vec = std_dist_in_W(:);
[lambda_ind, sol_ind] = ind2sub([nLambdas,  nStartPoints], 1:nLambdas*nStartPoints);
[std_dist_in_W_vec_sort, ind_sort]  = sort(std_dist_in_W_vec, 'ascend');

% show frame potential ordered by prev. sorting
frame_pot_W_vec = frame_pot_W(:);
% ordered by std
frame_pot_W_vec_sort = frame_pot_W_vec(ind_sort);


%% Sort matrices based on std and save
for i=1:length(ind_sort)
    l = lambda_ind(ind_sort(i));
    s = sol_ind(ind_sort(i));
    We_lambda_sorted(:,:,i) = We_lambda(:,:,s,l);
end
if 0
    commonFileName = sprintf('cv_%s_%s_sd_%d_lw_%d_startpoints_%d_sdpoints_%d', tagLoss, tagData, sd, nLambdas, numStartPoints, 0);
    filename_sorted = fullfile(dirData, [commonFileName '_std_ordered.mat']);
    save(filename_sorted, 'We_lambda_sorted', 'lambda_w_vals', 'std_dist_in_W_vec_sort', 'ind_sort', 'lambda_ind', 'sol_ind', 'frame_pot_W_vec_sort');
end

%% Std and frame potential plots

c = lines(3);
figure; plot(std_dist_in_W_vec_sort, '.-', 'linewidth', 2); axis tight; ylabel('std'); axis tight; % grid on;
printif(gcf, fullfile(dirFigs, sprintf('%s_%s', 'plot_std_ordered', commonFileName)), flag_print, false, false);

figure; plot(frame_pot_W_vec(ind_sort), '.-', 'linewidth', 2, 'Color', c(2,:)); ylabel('frame pot.'); axis tight;
printif(gcf, fullfile(dirFigs, sprintf('%s_%s', 'plot_frame_pot', commonFileName)), flag_print, false, false);

figure; plot(std_dist_in_W_vec_sort, frame_pot_W_vec(ind_sort), '.-', 'linewidth', 2, 'Color', c(3,:));
xlabel('std'); ylabel('frame pot.'); axis tight;
printif(gcf, fullfile(dirFigs, sprintf('%s_%s', 'plot_std_vs_frame_pot', commonFileName)), flag_print, false, false);

%% 50 'best' solutions ordered by std
flag_print = true;
h1 = figure('units','normalized','outerposition',[0 0 1 1]);
h2 = figure('units','normalized','outerposition',[0 0 1 1]);
for i=1:n_sol
    l = lambda_ind(ind_sort(i));
    s = sol_ind(ind_sort(i));
    Wel = reshape(We_lambda(:,:,s,l), d, kOrbitSize);
    Gel = Wel'*Wel;
    figure(h1); subaxis(5, n_sol/5, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Wel); axis equal; axis off;
    % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
    title(sprintf('(%d,%d)', l, s));
    figure(h2); subaxis(5, n_sol/5, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Gel); axis square; axis off;
    % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
    title(sprintf('(%d,%d)', l, s));
end
printif(h1, fullfile(dirFigs, sprintf('%s_%s', 'We_std_ordered', commonFileName)), flag_print, false, false); close(h1);
printif(h2, fullfile(dirFigs, sprintf('%s_%s', 'Ge_std_ordered', commonFileName)), flag_print, false, false); close(h2);



%% ************************************************************************
%% Additional/custom (CBMM Memo figures)
h1 = figure('units','normalized','outerposition',[0 0 1 1]);
h2 = figure('units','normalized','outerposition',[0 0 1 1]);

switch tagData
    
    case 'Pyritohedral_6';        
        for i=1:6
            l = lambda_ind(ind_sort(i));
            s = sol_ind(ind_sort(i));
            Wel = reshape(We_lambda(:,:,s,l), d, kOrbitSize);
            Gel = Wel'*Wel;
            figure(h1); subaxis(5, 3, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Wel); axis equal; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
            figure(h2); subaxis(1, 6, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Gel); axis square; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
        end
        
        printif(h1, fullfile(dirFigs, sprintf('%s_%s_sol_6', 'We_std_ordered', commonFileName)), flag_print, false, false); close(h1);
        printif(h2, fullfile(dirFigs, sprintf('%s_%s_sol_6', 'Ge_std_ordered', commonFileName)), flag_print, false, false); close(h2);
        
    case 'Dihedral_6';        
        for i=1:10
            l = lambda_ind(ind_sort(i));
            s = sol_ind(ind_sort(i));
            Wel = reshape(We_lambda(:,:,s,l), d, kOrbitSize);
            Gel = Wel'*Wel;
            figure(h1); subaxis(5, 5, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Wel); axis equal; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
            figure(h2); subaxis(1, 10, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Gel); axis square; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
        end
        
        printif(h1, fullfile(dirFigs, sprintf('%s_%s_sol_10', 'We_std_ordered', commonFileName)), flag_print, false, false); close(h1);
        printif(h2, fullfile(dirFigs, sprintf('%s_%s_sol_10', 'Ge_std_ordered', commonFileName)), flag_print, false, false); close(h2);
        
    case 'Cyclic_6'
        for i=1:20
            l = lambda_ind(ind_sort(i));
            s = sol_ind(ind_sort(i));
            Wel = reshape(We_lambda(:,:,s,l), d, kOrbitSize);
            Gel = Wel'*Wel;
            figure(h1); subaxis(5, 10, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Wel); axis equal; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
            figure(h2); subaxis(5, 10, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03); imagesc(Gel); axis square; axis off;
            % title(sprintf('(l,s,fp)=(%d,%d,%1.2f)', l, s, frame_pot_W_vec(ind_sort(i))));
            title(sprintf('(%d,%d)', l, s));
        end
        
        printif(h1, fullfile(dirFigs, sprintf('%s_%s_sol_20', 'We_std_ordered', commonFileName)), flag_print, false, false); close(h1);
        printif(h2, fullfile(dirFigs, sprintf('%s_%s_sol_20', 'Ge_std_ordered', commonFileName)), flag_print, false, false); close(h2);
        
end
