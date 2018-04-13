% Regularizer for multiple orbits

function reg = regW_mult(W, setm, k, s, E)

nGroups = unique(setm); % number of orbits in layer

if nargin<5
    E = kron(eye(k), ones(k));    
end
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

% sum over each groups/orbits
reg = 0;
for g=nGroups
    reg = reg + regW_fixed(W(:, setm==g), k, s, kE_term);
end

% reg. is scaled by number of orbits/average
reg = reg/length(nGroups);
