% Collect and merge in single .mat file individual experiment files from OM
%
% Each run corresponds to a single lambda and multiple solutions. Compose
% single We (of size M x N x solutions x lambdas)

% run: 1. om_min_regWComm_cv.m
%      2. load_merge_om_single_files.m
%      3. ...

% GE, CBMM/LCSL/MIT, gevang@mit.edu

clear; close all;

%*** Need to be set*******
dirData = '/media/gevang/Data/work/exp/orblearn/data'; % dir where mat data files are located
tagData = 'Pyritohedral_6'; % 'Pyritohedral_6'; 
lambda_vals = 10.^(0:7);
tagLoss = 'regWCommReLU_300';
%*************************

for l = 1:length(lambda_vals);
    filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_50_sdpoints_0.mat', tagLoss, tagData, lambda_vals(l)));
    load(filename_cv)
    We(:,:,:,l) = We_lambda;
end

% vars to save 
We_lambda = We;
lambda_w_vals = lambda_vals;

filename_cv = fullfile(dirData, sprintf('cv_%s_%s_sd_1_lw_%d_startpoints_50_sdpoints_0.mat', tagLoss, tagData, length(lambda_w_vals)));
save(filename_cv, 'We_lambda', 'lambda_w_vals', 'sd_points');
