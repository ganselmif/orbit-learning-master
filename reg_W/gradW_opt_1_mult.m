% Computes the gradient of regW_mult, i.e. 1/|G|\sum_g reg(W)

function dW = gradW_opt_1_mult(W, setm, k, s, Ik, E, CRt)

% create the auxiliary matrices
if nargin<5
    Ik = sparse(eye(k)); %Ik = eye(k);
    E = kron(Ik, ones(k));
    % end
    % if nargin<6
    [C, R] = gradW_opt_aux(k);
    CRt = R'*C';
end

nGroups = unique(setm); % number of orbits in layer
dW = [];
for g=nGroups
    ind_g = setm==g;
    dW = [dW; gradW_opt_1_fixed(W(:, ind_g), k, s, Ik, E, CRt)];
end

% scale gradient by number of orbits
dW = dW/length(nGroups);