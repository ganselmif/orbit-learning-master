% Import the permutations generators of a group from .txt file 

function P = Pmat(filename)

G = importdata(filename);

[m, n] = size(G); order = m/n;
for ind = 1:order
    P(:,:,ind) = G(n*(ind-1)+1:ind*n,:);
end
% P = reshape(G, [n n order]);
