% Plotting min objective function solutions for various lambda
%
% From: confusion_matrix_intra_inter.m 

clear; close all;
dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located
flag_print = true;
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/regWdist_lambda'; % dir where mat data files are located

tagLoss = 'regWCommReLU_300'; 
numStartPoints = 50;

tagData = 'Cyclic_6';
switch tagData
    case 'Cyclic_6';
        nLambdas = 8;        
        no_orb_show = 12;
        
    case 'Dihedral_6'
        nLambdas = 8;
        no_orb_show = 6;
        
    case 'Pyritohedral_6';
        nLambdas = 8;       
        no_orb_show = 3;
end

% Load solution 'orbit'
typeOrder = 'best';
switch typeOrder
    case 'stdorder'
        % filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0_std_ordered.mat', tagLoss, tagData, nLambdas, numStartPoints));
    otherwise
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_%d_sdpoints_0.mat', tagLoss, tagData, nLambdas, numStartPoints));
end
% Load validation orbits (pre-computed)
filename_db = fullfile(dirData, sprintf('imdb_%s_sds_20', tagData));

load(filename_db);
load(filename_cv);

%% Data: training set
sd = 1; % seed

Xr = X(:, ind_set==1, sd); ind_xr = ind_orbit(ind_set==1);
N = sum(ind_t==1); % [d, Nx] = size(Xr); kOrbitSize = Nx./N;

%% Data: validation set
% Xt = X(:, ind_set==3, sd);
Xv_all = X(:, ind_set==2, sd); ind_xv_all = ind_orbit(ind_set==2);
Tv = T(:, ind_t==2, sd); 

%% Subsample val
nOrbits = 50; % size(Xv_all, 2)
subScheme = 'decor';
s = rng; rng(0);
r = subSampleSet(Tv, nOrbits, subScheme);
rng(s);

% Tvr = Tv(:,r); figure; imagesc(project_unit_norm(Tvr)'*project_unit_norm(Tvr)); colorbar;

%% Subset of validation set used 
ind_orbit_val = r + min(ind_xv_all)-1;  
ind_points_from_orbit = ismember(ind_xv_all', ind_orbit_val', 'rows');
% sum(ind_points_from_orbit)

Xv = Xv_all(:, ind_points_from_orbit);
ind_xv = ind_xv_all(ind_points_from_orbit);

[d, Nv] = size(Xv); kOrbitSize = Nv./nOrbits;


clear X Xv_all ind_orbit filename_db sd_vec s dataGroupName r;


%% Distance function
s = 10;
dimRepVec = kOrbitSize;
% Auxiliary
k2 = 2; E = kron(eye(k2), ones(dimRepVec)); kE_term = (k2*E - 1) - 0.5*(k2-1)*eye(dimRepVec*k2);
kd = 2*dimRepVec; % number of comparison elements
c = kd*(kd+1)/2; % scaling constant = number of lower triangular elements


%% Inter-intra- orbit distances
[ind_in, ind_out] = indMatInterIntraOrbit(nOrbits, kOrbitSize);

ind_both = zeros(size(ind_in));
ind_both(ind_in==1) = 1;
ind_both(ind_out==1) = 2;


%% Vectorized use with pdist2 
distFun = @(XI,XJ)(regWdist1(XI, XJ, s, kE_term, c));

h1 = figure('units','normalized','outerposition',[0 0 1 1]);

for l=1:nLambdas
    dicType = sprintf('regw_lw_%d', l);
    
    W = We_lambda(:,:,1,l);
    
    %% Normalized dot-product with dictionary
    % unit-norm
    dotProdMat = project_unit_norm(W)'*project_unit_norm(Xv);
    
    D = zeros(Nv);
    
    for i=1:Nv
        % disp(i);
        D(i,i+1:end) = pdist2(dotProdMat(:,i)', dotProdMat(:,i+1:end)', distFun);
    end
    Ds = D + D';
    distMatOrbit = log(abs(Ds)); % log(Ds);
    
    k = kOrbitSize; nComp = no_orb_show;
        
    if flag_print
        [h2, h0] = imOrbitDistMatrix(distMatOrbit, ind_in, ind_out, kOrbitSize, no_orb_show); close(h0);
        commonFileName =  sprintf('%s_%s_StartPoints_50', tagData, tagLoss);        
        set(h2.CurrentAxes, 'Visible','off'); figure(h2);
        printif(h2, fullfile(dirFigs, sprintf('%s_%s_%d_val_%d_%s_log', commonFileName,'distMat', no_orb_show, nOrbits, dicType)), flag_print, false, false);
        close(h2);
    end
    
    figure(h1);
    subaxis(2, 8, l, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03);
    imagesc(distMatOrbit(1:k*nComp, 1:k*nComp)); axis square; axis off;
    title(sprintf('log(\\lambda) = %d', log10(lambda_w_vals(l))));    
end

if flag_print
    set(h1.CurrentAxes, 'Visible','off'); figure(h1);
    printif(h1, fullfile(dirFigs, sprintf('%s_%s_%d_val_%d_regw_lw_all_log', commonFileName,'distMat', no_orb_show, nOrbits)), flag_print, false, false);
    close(h1);
end



%% Ordered per std: need
if strcmp(typeOrder, 'stdorder')
    for j=1:8
        W = We_lambda_sorted(:,:,j);
        l = lambda_ind(ind_sort(j));
        s = sol_ind(ind_sort(j));
        
        %% Normalized dot-product with dictionary
        % unit-norm
        dotProdMat = project_unit_norm(W)'*project_unit_norm(Xv);
        
        D = zeros(Nv);
        
        for i=1:Nv
            % disp(i);
            D(i,i+1:end) = pdist2(dotProdMat(:,i)', dotProdMat(:,i+1:end)', distFun);
        end
        Ds = D + D';
        distMatOrbit = log(abs(Ds)); % log(Ds);
        
        % distMat(ind_both==0) = nan;
        % ind_both(ind_both==0) = nan;
        k = kOrbitSize; nComp = no_orb_show;
        
        subaxis(2, 8, j, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03);
        imagesc(distMatOrbit(1:k*nComp, 1:k*nComp)); axis square; axis off;
        title(sprintf('(%d,%d)', l, s));
        
        %% Distributions
        % hs = subaxis(2, 8, i, 'Spacing', 0.001, 'Padding', 0.005, 'Margin', 0.03);
        % nBins = 50;
        % hs = plotDistMatrix(log(distMatOrbit), ind_in, ind_out, nBins, hs);
        % title(sprintf('l=%d', lambda_w_vals(l)));
        % pause
    end
end

