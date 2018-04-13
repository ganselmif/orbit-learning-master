% Gradient of all-differences regularizer for a single group Wp (cross)

function dW = gradW_opt_1_cross(Wp, W, setm, p, k, s, Ik, E, CRt, Ct, CTt) % SCt, STCt)

if nargin<7
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
dW = 0;
for q=nGroups
    
    Wq = W(:, setm==q);          
     
    if q == p
        P1 = CRt; 
        Ipq = gradCrossTerm(Wp, Wq, E, k, s);
        % gradW_opt_1(Wq, k, s, Ik, E, P);
        dW = dW + kron(Ik, Wq)*(P1*Ipq); % parenthesis are needed for sparse gpuArrays
    else
        Ipq = gradCrossTerm(Wp, Wq, E, k, s);
        Iqp = gradCrossTerm(Wq, Wp, E, k, s);
        P1 = CTt;
        P2 = Ct;
       dW = dW + kron(Ik, Wq)*(P1*Ipq + P2*Iqp); % parenthesis are needed for sparse gpuArrays
    end            
    
end
c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
dW = dW/c;


function I = gradCrossTerm(Wp, Wq, E, k, s)
% Computing the part of the gradient coresponding to diag()'*t
% 
% See also gradW_opt_1

vecG = vec(Wp'*Wq);

dev = tril(bsxfun(@minus, vecG', vecG));
dev = -(2/s).*(dev.*exp(-((dev).^2)./s));
dev = (k*E - 1).*dev; % EC.*dev;

I = dev(~triu(ones(size(dev))));








