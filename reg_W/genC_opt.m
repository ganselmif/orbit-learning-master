% (Optimized) matrix C computation for regW. 
%
% Source: genC.m

function C = genC_opt(k)

% full matrix in memory -- will use this to pick up chunks of diff. size
k2 = k^2;
Id_full = sparse(eye(k2-1));
I_full = sparse(ones(k2,1));

C = cell(k2-1);

for i = 1:k2-1
    % Id = Id_full(1:i, 1:i);
    C{i} = [kron(I_full(1:k2-i,1), Id_full(i, 1:i)), - Id_full(1:k2-i, 1:k2-i)];
end

C = vertcat(C{:});