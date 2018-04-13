% Comparison of dictionaries using invariance in signature
%
% see also dispComparisonGD.m, makeSignatureDist.m

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function [d_in, d_out, Os] = ComparisonGD(filename, Ol, fix_seed, nTrials, filename_dif)

if nargin<4, nTrials = 1; end;
if nargin==2, fix_seed = true; end
if fix_seed, rng(42, 'twister'); end; % Fix seed for reproducible results

%% Type of 'random' vector sampling
sampleType = 'uball';

%% Orbit reference/generators
P = importPermutationsToMat(filename); % generators of G
[~, d, sizeOrbit] = size(P);
% index_orbit = ones(sizeOrbit, 1); % for pooling

%% Generate different templates (trials)
nSamplesOrbit = sizeOrbit; % number of orbit samples of v and t w.r.t. G

% Random vec to be transformed by G for testing signatures
t = randSampleVec(d, nTrials, sampleType);
% t = t./norm(t);

for c = 1:nTrials    
    % Sample at random
    indSamples = randperm(sizeOrbit, nSamplesOrbit);
    % indSamples = 1:order;
    
    for i = 1: length(indSamples)
        Ot(:,i,c) = P(:,:,indSamples(i))*t(:,c);
    end
end
% Each transformed template is an 'input' to be evaluated
Ott = reshape(Ot, [d, nSamplesOrbit*nTrials]);
% index_Ot = kron(1:nTrials, ones(1, n)); % test orbit index


%% Input can be a single dictionary or a tensor of different dictionaries
[d, sizeDic, nDics] = size(Ol);
if nDics>1
    % Multiple solutions/points/dictionaries
    % Collapse in single matrix
    Ol = reshape(Ol, [d, sizeDic*nDics]);  
end

% orbit index (for pooling)
index_orbit = kron(1:nDics, ones(1, sizeDic))'; % for pooling

%% 1. Random orbit
Or = randSampleVec(d, sizeDic*nDics, sampleType);
% Or = bsxfun(@rdivide, Or, sqrt(sum(Or.^2)));

%% 2. Orbit with same group/symmetry
v = randSampleVec(d, nDics, sampleType);
% v = v./norm(v);
for c = 1:nDics
    for i = 1:sizeOrbit %generates the orbit of v and t w.r.t. G
        Os(:,(c-1)*sizeDic + i) = P(:,:,i)*v(:,c);
    end
end

if nargin==5;
    %% 3. Orbit with different symmetry
    Pd = importPermutationsToMat(filename_dif); % generators of G
    [~, ~, sizeOrbitDif] = size(Pd);
    for c = 1:nDics
        for i = 1:sizeOrbitDif %generates the orbit of v and t w.r.t. G
            Od(:,(c-1)*sizeOrbitDif + i) = Pd(:,:,i)*v(:,c);
        end
    end
    index_orbit_dif = kron(1:nDics, ones(1, sizeOrbitDif))'; % for pooling
    clear Pd sizeOrbitDif
end

%% Util matrices for in-orbit and Out-of-orbit distributions
[ind_in, ind_out] = indMatInterIntraOrbit(nTrials, nSamplesOrbit); 

%% Distance computation
poolMethodNames = {'max', 'l2', 'lp', 'meanrelu', 'rms', 'mean', 'hist', 'moments', 'centmom', 'sumstats', 'cdf', 'histc'};
distMethodNames = {'euclidean', 'cosine', 'seuclidean', 'chebychev', 'minkowski'};

% Pool and distance method
poolMethod = poolMethodNames{4};
distMethod = distMethodNames{1};
% typeSignature = 'single';

% Same symmetry
[ds_i, ds_o] = makeSignatureDist(Ott, Os, poolMethod, distMethod, index_orbit, ind_in, ind_out);
% Learned symmetry
[dl_i, dl_o] = makeSignatureDist(Ott, Ol, poolMethod, distMethod, index_orbit, ind_in, ind_out);
% Random
[dr_i, dr_o] = makeSignatureDist(Ott, Or, poolMethod, distMethod, index_orbit, ind_in, ind_out);

%% Concatenate outputs
d_in = [ds_i, dl_i, dr_i];
d_out = [ds_o, dl_o, dr_o];

%% Add dif. symmetry if asked
if nargin==5;
    % Orbit with different symmetry
    [dd_i, dd_o] = makeSignatureDist(Ott, Od, poolMethod, distMethod, index_orbit_dif, ind_in, ind_out);
    d_in = [d_in, dd_i];
    d_out = [d_out, dd_o];
end

