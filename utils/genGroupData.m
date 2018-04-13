% Generate N orbits for a known group for sampled or provided templates 
% 
% X = genGroupData(filenameGroup, [], T) generates all transformations in T
% wrt. the group generators stored in "filename.txt"
%
% X = genGroupData(filename, N) generates T by uniformly sampling from unit
% simplex or unit ball (default).

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function [X, T] = genGroupData(filenameGroup, numT, T)

P = importPermutationsToMat(filenameGroup);
[dimGroup, ~, orderGroup] = size(P);

if nargin<3
    % templates are not provided
    samplType = 'uball'; % random vector sampling type
    T = randSampleVec(dimGroup, numT, samplType);
elseif numel(T)==1
    % not provided but T subsets needed, i.e. of controlled variance
    samplType = 'mball';
    T = randSampleVec(dimGroup, numT, samplType, T);
else
    numT = size(T, 2);
end

% apply group transformation matrices on template matrix
for indMat = 1:orderGroup
    X(:, indMat, :) = P(:, :, indMat)*T;
end
% re-order orbit-wise
X = reshape(X, [dimGroup, orderGroup*numT]);