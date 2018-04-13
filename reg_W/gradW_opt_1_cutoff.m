% Optimized computation for gradW with cutoff

function dW = gradW_opt_1_cutoff(W, k, s, Ik, E, CRt)

if nargin<4
    Ik = sparse(eye(k)); %Ik = eye(k);
    E = kron(Ik, ones(k));
end

% G = W'*W;
vecG = vec(W'*W);

% sign is important: has to have the opposite sign than the regularizer!?
diffG = tril(bsxfun(@minus, vecG', vecG));
% dev = tril(bsxfun(@minus, vecG, vecG'));
%clear vecG;

dev = -(2/s).*(diffG.*exp(-((diffG).^2)./s));

% EC=ones(k^2);
dev = (k*E - 1).*dev; % EC.*dev;

% tic; I = dev(itril(size(dev),-1)); toc
% same computation (& much faster) without calling itril
I = dev(~triu(ones(size(dev))));
%clear E dev;

% R = sparse(eye(k^2) + genT_opt(k));
% C = genC_opt(k);
if nargin<6
    % create the auxiliary matrices
    [C, R] = gradW_opt_aux(k);
    CRt = R'*C';
end

% P = C*R; % this multiplication was killing it!
%clear R C;
c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
dW = kron(Ik, W)*(CRt*I)/c; % parenthesis are needed for sparse gpuArrays

%% Adding cut-off function

reg = (k.*E - 1).*exp(-((diffG).^2)./s);
reg = sum(sum(reg))./c;

% e = 10^-6; a = 10^10;
% reg = sigm(reg - e, a);
e = 10; a = 100; 
y = exp(a*(reg) - e);

dW = y*(1/(1 + y))*a*dW;



