% Import the permutations generators of a group stored in "filename".

function P = importPermutationsToMat(filename)

G = importdata(filename);

[l, n] = size(G);
orderGroup = l/n;

% P = reshape(G', [n n orderGroup]);
for indMat = 1:orderGroup
    P(:, :, indMat) = G(n*(indMat-1)+1:indMat*n, :);
end