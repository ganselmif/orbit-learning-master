% Same as regW_fixed but minimizing reg over a set of sigma values also.  

function [reg, reg_s, smin] = regW_fixed_min_s(W, k, s, E)

G = W'*W;
vecG = vec(G); 

difG = bsxfun(@minus, vecG', vecG);

if nargin<4
    E = kron(eye(k), ones(k));    
end

%% (kE-1): the additional term removes the double contribution of the diagonal elements
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2)); 

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
for i=1:length(s)
    reg_c = tril(kE_term.*exp(-((difG).^2)./s(i)));
    % reg = (1-k.*E).*reg; % + reg;      
    reg_s(i) = sum(sum(reg_c))./c;
end
[reg, i] = min(reg_s); % minimize over sigma
smin = s(i); 

%reg=sum(sum((-k).*E.*exp(-((E.*reg).^2)./s) + EC.*exp(-((EC.*reg).^2)./s)));
