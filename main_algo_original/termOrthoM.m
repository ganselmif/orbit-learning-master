% Compute the orthogonality constraint

function V = termOrthoM(M, Ik)

k = size(M,3);
if nargin<2, Ik = eye(k); end
Mt = permute(M, [2,1,3]);

for indMat = 1:k
    V(indMat) = norm(M(:,:,indMat)*Mt(:,:,indMat)-Ik, 'fro').^2;
end
V = sum(V);
% clear Mt;
% norm((MMt-repmat(eye(k), 1, 1, k)).^2,'fro')