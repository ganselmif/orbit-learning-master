% Optimized computation for gradW

function dW = gradW_opt(W, k, s)

% G = W'*W;
vecG = vec(W'*W);

% sign is important: has to have the opposite sign than the regularizer!?
dev = tril(bsxfun(@minus, vecG', vecG));
% dev = tril(bsxfun(@minus, vecG, vecG'));
%clear vecG;

dev = -(2/s).*(dev.*exp(-((dev).^2)./s));

Ik = sparse(eye(k)); %Ik = eye(k);
E = kron(Ik, ones(k));
% EC=ones(k^2);
dev = (k*E - 1).*dev; % EC.*dev;

% tic; I = dev(itril(size(dev),-1)); toc
% same computation (& much faster) without calling itril
I = dev(~triu(ones(size(dev))));
%clear E dev;

R = sparse(eye(k^2) + genT_opt(k));
C = genC_opt(k);
% create the auxiliary matrices
% [C, R] = gradW_opt_aux(k);

% P = C*R; % this multiplication was killing it!
%clear R C;

dW = kron(Ik, W)*R'*C'*I;
