% Gradient of all-differences regularizer for multiple orbits 
% (with cross-orbit terms)

function dW = gradW_opt_1_mult_cross(W, setm, k, s, Ik, E, CRt, Ct, CTt)

if nargin<5
    Ik = sparse(eye(k)); %Ik = eye(k);
    E = kron(Ik, ones(k));
    % create the auxiliary matrices
    [C, R] = gradW_opt_aux(k);
    Ct = C'; 
    CRt = R'*Ct;
    CTt = genT_opt(k)*Ct;
    % SCt = S*C';
    % STCt = S*genT_opt(k)*C';
end

nGroups = unique(setm); % number of orbits in layer
dW = [];
for p=nGroups
    dW = [dW; gradW_opt_1_cross(W(:, setm==p), W, setm, p, k, s, Ik, E, CRt, Ct, CTt)];
end
