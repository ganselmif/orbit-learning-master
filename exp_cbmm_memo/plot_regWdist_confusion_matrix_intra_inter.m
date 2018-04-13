% From: confusion_matrix_intra_inter

clear; close all;
dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located
% dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/regWdist2'; % dir where mat data files are located
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/regWdist2_sigproc_figs';

flag_print = true;
lw = 6;
fn = 24;

nReg = 300;
nOrbits = 50; % size(Xv_all, 2)

tagLoss = sprintf('regWCommReLU_%d', nReg); 
tagData = 'Pyritohedral_6';

tagDic = 'regw';

% *************************************************************************
% Load data
% *************************************************************************
numStartPoints = 50;
switch tagData
    case 'Cyclic_6';
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, sprintf('imdb_%s_sds_20', tagData));
        
        no_orb_show = 12;
        
    case 'Dihedral_6'
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, sprintf('imdb_%s_sds_20', tagData));
        
        no_orb_show = 6;
        
    case 'Pyritohedral_6';
        nLambdas = 8;
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, sprintf('imdb_%s_sds_20', tagData));
        
        no_orb_show = 3;
end
load(filename_cv);
load(filename_db);


% *************************************************************************
% Data: train, validation and subsampling
% ************************************************************************

%% Data: training set
sd = 1; % seed

Xr = X(:, ind_set==1, sd); ind_xr = ind_orbit(ind_set==1);
N = sum(ind_t==1); % [d, Nx] = size(Xr); kOrbitSize = Nx./N;

%% Data: validation set
% Xt = X(:, ind_set==3, sd);
Xv_all = X(:, ind_set==2, sd); ind_xv_all = ind_orbit(ind_set==2);
Tv = T(:, ind_t==2, sd); 

%% Subsample validation set 
s = rng; rng(0);
subScheme = 'decor'; % sample most decorrelated orbits 
r = subSampleSet(Tv, nOrbits, subScheme);
rng(s);

% Plot template correlation
if 0
    Tvr = Tv(:,r);
    figure; imagesc(project_unit_norm(Tvr)'*project_unit_norm(Tvr)); axis image; colorbar;
end

%% Subset of validation set used 
ind_orbit_val = r + min(ind_xv_all)-1;  
ind_points_from_orbit = ismember(ind_xv_all', ind_orbit_val', 'rows');
% sum(ind_points_from_orbit)

Xv = Xv_all(:, ind_points_from_orbit);
ind_xv = ind_xv_all(ind_points_from_orbit);

[d, Nv] = size(Xv); kOrbitSize = Nv./nOrbits;

clear X Xv_all ind_orbit filename_db sd_vec s dataGroupName r;


% *************************************************************************
% Representation learning
% ************************************************************************

%% Dictionary/representation type
switch tagDic

    case 'regw'
        %% Learned orbit
        % pick single lambda, best solution
        l = round(log10(kOrbitSize^3*N));
        % l = round(log10((d*d*N*kOrbitSize)));
        tagDic = sprintf('regw_lw_%d', l);  
        W = We_lambda(:,:,1,l);        
        
    case 'symg'
        %% True orbit (from val set)
        r = randi(nOrbits); id_xv = unique(ind_xv); r = id_xv(r); 
      
        W = Xv(:, ind_xv==r);
        clear r id_xv;
           
    case 'eyew'
        %% Distance on raw pixels (map is I)
        W = eye(d, kOrbitSize); % eye(d); %, kOrbitSize);
        
    case 'rand'
        sampleType = 'uball';
        %% Random orbit
        W = randSampleVec(d, kOrbitSize, sampleType);
        
    case 'pca'
        %% Simple PCA
        [V , ~] = eigpca(Xr');
        W = V(:, 1:min(d, kOrbitSize)-1); % projection on k first eigenvectors
        
    case 'ica'
        %% Simple ICA
        [V, D] = eigpca(Xr'); rdim = size(V, 1);
        Vw = diag(diag(D).^-0.5)*V'; % whitening matrix  sqrt(D)\E';
        % Vd = V*diag(sqrt(diag(D)));  % dewhiteningMatrix
        Xw = Vw(1:rdim,:)*Xr;  % whitened data
        
        W = ica(Xw, rdim, 500, 10e-4);
        % [~, ~, W1] = fastica(X, 'approach', 'defl', 'pcaE', V, 'pcaD', diag(D), 'whiteSig', Xw', 'whiteMat', Vw, 'dewhiteMat', Vd);
        % transform back to original space from whitened space and compute A using pseudoinverse (inverting canonical preprocessing is tricky)
        Aica = pinv(W*Vw(1:rdim,:));
        
        % weight matrix
        W = Aica(:, 1:min(d, kOrbitSize)-1);
        
    case 'ksvd'
        %% Sparse dictionary learning with KSVD
        optKSVD.data = project_unit_norm(Xr);
        optKSVD.Tdata = 1;
        optKSVD.codemode = 'sparsity';
        % optKSVD.Edata = 10^-1;
        optKSVD.dictsize = kOrbitSize;
        optKSVD.iternum = 50;
        optKSVD.memusage = 'high';
                
        [W, A, err] = ksvd(optKSVD,'');
        
    case 'ae'
        %% Autoencoder
        ae = trainAutoencoder(Xr, kOrbitSize,...
            'MaxEpochs', 1000, ...
            'L2WeightRegularization', 1,...
            'SparsityRegularization', 3,...
            'SparsityProportion', 0.05,...
            'EncoderTransferFunction','logsig',... 'satlin'
            'DecoderTransferFunction','purelin',...
            'ScaleData', true, ...
            'ShowProgressWindow', false);

        W = ae.EncoderWeights';
        % mse(Xr - predict(ae, Xr))
        % dotProdMat = encode(ae, Xv);
        
end

h1 = figure; imagesc(project_unit_norm(W)); axis equal; axis off;% colorbar;
h2 = figure; imagesc(project_unit_norm(W)'*project_unit_norm(W)); axis equal; axis off; % colorbar;
if 0 % ~flag_print
    commonFileName =  sprintf('%s_%s_StartPoints_50', tagData, tagLoss);
    
    printif(h1, fullfile(dirFigs, sprintf('%s_%s_%s', commonFileName,'W', dicType)), flag_print, false, false); close(h1);  
    printif(h2, fullfile(dirFigs, sprintf('%s_%s_%s', commonFileName,'G', dicType)), flag_print, false, false); close(h2);    
end


%% regw distance function
s = 10;
[distMatOrbit, dotProdMat] = regWdistWrapper(W, Xv, s);

%% Inter-intra- orbit distances
[ind_in, ind_out] = indMatInterIntraOrbit(nOrbits, kOrbitSize);
[h2, h0] = imOrbitDistMatrix(log(distMatOrbit), ind_in, ind_out, kOrbitSize, no_orb_show); close(h0);

%% Distributions
nBins = 50;
% h4 = figure('units','normalized','outerposition',[0 0 0.5 0.7]);
h4 = plotDistMatrix(log10(distMatOrbit), ind_in, ind_out, nBins, [], lw, fn);% nComp)

if flag_print
    commonFileName =  sprintf('%s_%s_StartPoints_50', tagData, tagLoss);
    
    set(h2.CurrentAxes, 'Visible','off'); figure(h2);
    printif(h2, fullfile(dirFigs, sprintf('%s_%s_%d_val_%d_%s_log', commonFileName,'distMat', no_orb_show, nOrbits, tagDic)), flag_print, false, false);
    close(h2);
    
    figure(h4); grid off;
    
    ylim([0 0.090001])
    xlim([-20 0])
    % xlabel('$$\log(r(W^T[x,x])$$', 'fontsize', fn, 'Interpreter','latex')
    set(h4.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ... %'YMinorTick', 'on', ...
        'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'YTick', 0:0.02:0.10, 'XTick', -20:5:0, 'LineWidth', 1); 
    
    % set(findall(h4, '-property', 'FontSize'), 'FontSize', fn)
    printif(h4, fullfile(dirFigs, sprintf('%s_%s_val_%d_%s_log', commonFileName, 'distDist', nOrbits, tagDic)), flag_print, true, true);
    close(h4);
end

%% MDS visualization of redgist
if 0
    % close all
    no_dims = 2;
    typeVis = 'mds';
    switch typeVis
        
        case 'mds'
            %% Classic MD scaling
            [featOrbitMDS, ep] = cmdscale(distMatOrbit, no_dims);
    end
    featOrbitMDSZnorm = zscore(featOrbitMDS); % standardize scores in the 2 dimensions
    y = ind_xv; % index of orbit
    
    fn = 14; markersize = 40;
    h5 = figure; scatter(featOrbitMDSZnorm(:,1), featOrbitMDSZnorm(:,2), markersize, y, 'filled'); %, linewidth);
    % myscatter(featOrbitMDS(:,1:no_dims), y, [],  arrayfun(@num2str, 1:nOrbits, 'Uniform', false), fn);
    % axis equal; set(gca, 'box', 'off', 'zticklabel',[], 'xticklabel',[], 'yticklabel',[]); %, 'fontsize', fn); %, 'FontName', 'Helvetica')
    axis tight;
    set(gca, 'box', 'off', 'fontsize', fn, 'FontName', 'Helvetica', 'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3])
    set(gca,'XTickLabel',num2str(get(gca,'XTick').'))
    
    if flag_print
        commonFileName =  sprintf('%s_%s_StartPoints_50', tagData, tagLoss);
        printif(h5, fullfile(dirFigs, sprintf('%s_%s_val_%d_%s', ...
            commonFileName, 'mds', nOrbits, dicType)), true, false, false);
    end
    
end


if 0
    %% Pixel distance and distributions
    distMatPixel = pdist2(Xv', Xv', 'cosine');
    [h1, h0] = imOrbitDistMatrix(distMatPixel, ind_in, ind_out, kOrbitSize, no_orb_show); close(h0);
    h3 = plotDistMatrix(distMatPixel, ind_in, ind_out, nBins);% nComp)
    
    if flag_print
        commonFileName =  sprintf('%s_%s_StartPoints_50', tagData, tagLoss);
        
        set(h1.CurrentAxes, 'Visible','off'); figure(h1);
        printif(h1, fullfile(dirFigs, sprintf('%s_%s_%d_val_%s_cos', commonFileName, 'distMat', no_orb_show, 'pixel')), flag_print, false, false);
        close(h1);
        
        figure(h3); grid off;
        set(h3.CurrentAxes, 'FontName', 'Helvetica', 'Box', 'off', 'TickDir', 'out', ... % 'TickLength', [.02 .02], 'YMinorTick', 'on', ...
            'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
            'LineWidth', 2);
        set(findall(h3, '-property', 'FontSize'), 'FontSize', 14)
        printif(h3, fullfile(dirFigs, sprintf('%s_%s_val_%s_cos', commonFileName, 'distDist', 'pixel')), flag_print, true, true);
        close(h3);
    end
end

% figure; imagesc(Ds(1:6*5, 1:6*5)); axis square; colorbar;
% figure; imagesc(Da(1:6*5, 1:6*5)); axis square; colorbar;
