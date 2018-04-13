% Low, off-diagonal regularizer

function reg = regW_nz(W, k, s)

% k = size(W, 2);
% s = 0.001;

G = W'*W + eye(size(W, 2))*realmax;

vecG = vec(G);
% vecI = vec(ones(k) - eye(k));
% reg = exp(-((vecG.*vecI).^2)./s);
reg = exp(-((vecG).^2)./s);
reg = sum(sum(reg))/k^2;