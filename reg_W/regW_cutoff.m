function reg = regW_cutoff(W, k, s, E)

% k = size(W, 2);
G = W'*W;
vecG = vec(G); 
% difG = tril(bsxfun(@minus, vecG, vecG'));
difG = tril(bsxfun(@minus, vecG', vecG));

if nargin<4
    E = kron(eye(k), ones(k));
end
%EC = ones(k^2);
reg = (k.*E - 1).*exp(-((difG).^2)./s);
% reg = (1-k.*E).*reg; % + reg;

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
reg = sum(sum(reg))./c;
%reg=sum(sum((-k).*E.*exp(-((E.*reg).^2)./s) + EC.*exp(-((EC.*reg).^2)./s)));

%% Adding cut-off function

% e = 10^-6; a = 10^10;
% reg = sigm(reg - e, a);
e = 10; a = 100; 
reg = log(1 + exp(a*(reg) - e));



