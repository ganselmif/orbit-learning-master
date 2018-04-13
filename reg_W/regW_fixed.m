% Orbit regularization (default/fixed)
% 
% %% Example/Tests:
% W = genGroupData('CyclicGroup6.txt', 1);
% k = size(W, 2); s = 0.001;
% regW_fixed(W, k, s)
%
% %% Gradient test
% W = rand(k, k); % use random matrix
% testgradW(W, s, 5);

function reg = regW_fixed(W, k, s, kE_term) % , mask_b)

% G = W'*W; 
G = gramMat(W); % linear feature map Gramian
vecG = vec(G);
difG = bsxfun(@minus, vecG', vecG);

if nargin<4
    E = kron(eye(k), ones(k));
    
    %% (kE-1): the additional term removes the double contribution of the diagonal elements
    kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
end

reg_c = tril(kE_term.*exp(-((difG).^2)./s));

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
reg = sum(sum(reg_c))./c;


%% buggy version

% % difG = tril(bsxfun(@minus, vecG, vecG'));
% difG = tril(bsxfun(@minus, vecG', vecG));
%
% % difG = bsxfun(@minus, vecG', vecG));
%
% % difG(triu(ones(size(difG)))==1) = nan;
%
% if nargin<4
%     E = kron(eye(k), ones(k));
%     mask_b = tril(ones(k^2));
% end
% %EC = ones(k^2);
%
% reg_c = ((k*E - 1).*mask_b).*exp(-((difG).^2)./s);
% % reg = (1-k.*E).*reg; % + reg;
%
% c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
% reg = sum(sum(reg_c))./c;
% %reg=sum(sum((-k).*E.*exp(-((E.*reg).^2)./s) + EC.*exp(-((EC.*reg).^2)./s)));
