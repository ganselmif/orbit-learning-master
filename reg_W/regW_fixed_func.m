% Orbit regularization (default/fixed) with generic (nonlinear) Gramian
% 
% %% Example/Tests:
% W = genGroupData('CyclicGroup6.txt', 1);
% k = size(W, 2); s = 0.001;
%
% regW_fixed_func(W, k, s)
% regW_fixed_func(W, k, s, [], 'relu')
% regW_fixed_func(W, k, s, [], 'pow', 3)
%
% See also: gramMat.m gradW_opt_1_fixed_func.m script_test_reg_grad.m

% Taken from: regWh.m
% To-Do: Merge with regW_fixed.m

function reg = regW_fixed_func(W, k, s, kE_term, varargin) %typeFunc, p)

% if varargin==0, typeFunc = 'linear'; end % defaults to linear
G = gramMat(W, varargin{:}); 
vecG = vec(G);
difG = bsxfun(@minus, vecG', vecG);

if nargin<4 || isempty(kE_term)
    E = kron(eye(k), ones(k));
    kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
end
reg_c = tril(kE_term.*exp(-((difG).^2)./s));

c = k^2*(k^2-1)/2; % scaling constant = number of lower triangular elements
reg = sum(sum(reg_c))./c;
