% Generate S matrix for all-differences regularizer with multiple orbtis and cross-terms

function S = genS(k)

% Z = zeros(k);
S = cell(k,k); [S{:}] = deal(zeros(k));
for i=1:k
    for j=1:k
        % S{i,j} = Z;
        S{i,j}(i,j) = 1;
    end
end
S = sparse(cell2mat(S));