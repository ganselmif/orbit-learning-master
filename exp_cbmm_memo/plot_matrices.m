% Matrices of solutions and Gramians

% run:
% 1. om_min_regWCommReLU_cv

flag_print  = false;
dirFigs = '/media/gevang/Data/work/exp/orblearn/figs';
dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located
% dirFigs = '/media/gevang/Data/work/exp/orblearn/figs/regWdist1'; % dir where mat data files are located

tagLoss = 'regWCommReLU_100';
tagData = 'Dihedral_6';
commonFileName =  sprintf('%s_%s_startpoints_%d', tagLoss, tagData, 50);

switch tagData
    case 'Cyclic_6';
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_5_startpoints_50_sdpoints_0.mat', tagLoss, tagData));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Cyclic_6_sds_20');
        
    case 'Dihedral_6'
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_5_startpoints_50_sdpoints_0.mat', tagLoss, tagData));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Dihedral_6_sds_20');
        
    case 'Pyritohedral_6';
        % Load solution 'orbit'
        filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_6_startpoints_50_sdpoints_0.mat', tagLoss, tagData));
        % Load validation orbits (pre-computed)
        filename_db = fullfile(dirData, 'imdb_Pyritohedral_6_sds_20');
end
load(filename_cv)

d = size(We_lambda, 1);
kOrbitSize = size(We_lambda, 2);

%% All solutions for single lambda
h1 = figure('units','normalized','outerposition',[0 0 1 1]);
h2 = figure('units','normalized','outerposition',[0 0 1 1]);
for i=1:size(We_lambda,3)
    Wel = reshape(We_lambda(:,:,i,end), d, kOrbitSize);
    Gel = Wel'*Wel;
    
    % frame potential: sum of off-diagonal elements
    frame_pot_W(i) = sum(sum((normdotprod(Wel, Wel) - eye(kOrbitSize)).^2));
    
    figure(h1); subaxis(5, 10, i, 'Spacing', 0.001, 'Padding', 0, 'Margin', 0); imagesc(Wel); axis equal; axis off;
    figure(h2); subaxis(5, 10, i, 'Spacing', 0.001, 'Padding', 0, 'Margin', 0); imagesc(Gel); axis square; axis off;
end
printif(h1, fullfile(dirFigs, sprintf('%s_%s', 'We', commonFileName)), flag_print, false, false); close(h1);
printif(h2, fullfile(dirFigs, sprintf('%s_%s', 'Ge', commonFileName)), flag_print, false, false); close(h2);

% frame potential plot
figure; plot(1:50, frame_pot_W, 'o-', 'linewidth', 2); axis tight;
xlabel('solution index (best to worst)'); ylabel('frame potential');

%% Best solution/best lambda

load(filename_db); sd = 1;

% training set
Xr = X(:, ind_set==1, sd);
ind_xr = ind_orbit(ind_set==1);
[d, Nx] = size(Xr);
XXt = Xr*Xr'/Nx;

We = reshape(We_lambda(:,:,1,end), d, kOrbitSize);

Ge = We'*We;
GXe = We'*(XXt)*We;
Ce = abs(comm(XXt,We*We'));

h1 = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,3,1); imagesc(We); colorbar; title('W'); % axis equal; axis off;
subplot(2,3,2); imagesc(Ge); colorbar; title('W^TW'); axis square
% subplot(2,3,3); imagesc(GXe); colorbar; title('Gramian XtW'); axis square
subplot(2,3,3); imagesc(Ce); colorbar; title('Comm(XXt,WWt)'); axis square

%reference
Xr_o(:,:,1) = Xr(:, ind_xr==1); % Xr_o(:,:,2) = Xr(:, ind_xr==2);
G = Xr_o(:,:,1)'*Xr_o(:,:,1);
GX = Xr_o(:,:,1)'*(XXt)*Xr_o(:,:,1);
C = abs(comm(XXt,Xr_o(:,:,1)*Xr_o(:,:,1)'));

subplot(2,3,4); imagesc(Xr_o(:,:,1)); colorbar; title('Weights'); % axis square
subplot(2,3,5); imagesc(G); colorbar; title('Gramian'); axis square
% subplot(2,3,6); imagesc(GX); colorbar; title('Gramian XtW'); axis square
subplot(2,3,6); imagesc(C); colorbar; title('Comm(XXt,WWt)'); axis square


%% Save individual figures
flag_print  = true;

h1 = figure; imagesc(We); axis equal; axis off; 
h2 = figure; imagesc(Ge); axis equal; axis off;
h3 = figure; imagesc(Ce); axis equal; axis off;

printif(h1, fullfile(dirFigs, sprintf('%s_%s', 'We', commonFileName)), flag_print, false, false);
printif(h2, fullfile(dirFigs, sprintf('%s_%s', 'Ge', commonFileName)), flag_print, false, false);
% printif(h3, fullfile(dirFigs, sprintf('%s_%s', 'GXe', commonFileName)), flag_print, false, true);
printif(h3, fullfile(dirFigs, sprintf('%s_%s', 'Ce', commonFileName)), flag_print, false, false);


h4 = figure; imagesc(Xr_o(:,:,1)); axis equal; axis off;
h5 = figure; imagesc(G); axis equal; axis equal; axis off;
h6 = figure; imagesc(GX); axis equal; axis equal; axis off;

printif(h4, fullfile(dirFigs, sprintf('%s_%s', 'X', commonFileName)), flag_print, false, false);
printif(h5, fullfile(dirFigs, sprintf('%s_%s', 'G', commonFileName)), flag_print, false, false);
% printif(h6, fullfile(dirFigs, sprintf('%s_%s', 'GX', commonFileName)), flag_print, false, true);
printif(h6, fullfile(dirFigs, sprintf('%s_%s', 'C', commonFileName)), flag_print, false, false);


% Templates generating training set
Tr = T(:, unique(ind_xr), 1); t = 100;
h = figure; imagesc(Tr(:,1:t)); axis equal; axis off; box off;
printif(h, fullfile(dirFigs, sprintf('%s_%d_%s', 'T', t, commonFileName)), flag_print, false, false);