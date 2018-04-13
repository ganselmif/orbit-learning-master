% Calculate the permutation constraint 
% previously Mck.m

function mc = termPermuM(M) %, v)

mc = (sum(sum((sum(M, 2) - 1).^2)) + sum(sum((sum(M, 1) - 1).^2)));

% k = length(v);
% for indMatrix = 1 : k
%     mc(indMatrix) = norm(M(:,:,indMatrix)*v-v).^2 + norm((M(:,:,indMatrix))'*v-v).^2;
% end
% mc = sum(mc);