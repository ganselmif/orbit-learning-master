% All-differences regularizer for multiple orbits (using also cross terms)

function [reg, pair_reg] = regW_mult_cross(W, setm, k, s, E)

if nargin<5
    E = kron(eye(k), ones(k));
end

nGroups = unique(setm); % number of orbits in layer
pair_reg = zeros(k, k);
if isa(W, 'gpuArray');
    pair_reg = gpuArray(pair_reg);
end

for p=nGroups
    for q=nGroups
        Wp = W(:, setm==p);
        Wq = W(:, setm==q);
        % s = s + 1;
        %Gc(:,:,s) = Wdg1'*Wdg2; % Gramian
        pair_reg(p,q) = regW_cross(Wp, Wq, k, s, E);
    end
end
reg = sum(sum(pair_reg));



