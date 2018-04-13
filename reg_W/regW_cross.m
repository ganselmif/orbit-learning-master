% All-differences regularizer using different W1 and W2 (cross)

function reg = regW_cross(W1, W2, k, s, E)

% k = size(W, 2);
G = W1'*W2;
vecG = vec(G); 
% difG = tril(bsxfun(@minus, vecG, vecG'));
difG = tril(bsxfun(@minus, vecG', vecG));

if nargin<5
    E = kron(eye(k), ones(k));
end
%EC = ones(k^2);
reg = (k.*E - 1).*exp(-((difG).^2)./s);
% reg = (1-k.*E).*reg; % + reg;

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
reg = sum(sum(reg))./c;
%reg=sum(sum((-k).*E.*exp(-((E.*reg).^2)./s) + EC.*exp(-((EC.*reg).^2)./s)));
