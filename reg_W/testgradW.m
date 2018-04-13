% Compare analytic vs. numerical gradient for different regularizers
%
% % Example
% k = 20; W = rand(k, k); % use random matrix
% testgradW(W, s, 5); % linear 
% testgradW(W, s, 6); % relu feature map
% testgradW(W, s, 6, 'pow', 2); % power 2 feature map
%
% See also: script_test_reg_grad.m

function [diff, time_n] = testgradW(W, s, typeReg, varargin)
e = 10^-9;
if nargin<3, typeReg = 5; end
if nargin<2, s = 0.001; end

[d, k] = size(W);

% Auxiliary functions
Ik = sparse(eye(k));

% Type of regularizer (and gradient to check)
switch typeReg
    case 1
        regfun = @regW;
        gradfun = @gradW_opt_1; % gradW_opt; % gradW_opt_1
        E = kron(eye(k), ones(k));
        [C, R] = gradW_opt_aux(k);
        CRt = R'*C';
        varargin_grad = {Ik, E, CRt};
        varargin_reg = {E};
    case 2
        % version with square on the regularizer
        regfun = @regW_2;
        gradfun = @gradW_opt_2;
        E = kron(eye(k), ones(k));
        [C, R] = gradW_opt_aux(k);
        CRt = R'*C';
        varargin_grad = {Ik, E, CRt};
        varargin_reg = {E};
    case 3
        % non-zero regularizer
        regfun = @regW_nz;
        gradfun = @gradW_nz;
        R = sparse(eye(k^2) + genT_opt(k));
        varargin_grad = {Ik, R};
        varargin_reg = {};
    case 4
        % with cut-off
        regfun = @regW_cutoff;
        gradfun = @gradW_opt_1_cutoff;
        E = kron(eye(k), ones(k));
        [C, R] = gradW_opt_aux(k);
        CRt = R'*C';
        varargin_grad = {Ik, E, CRt};
        varargin_reg = {E};
        
    case 6
        % regularizer with nonlinear feature maps
        regfun = @regW_fixed_func;
        gradfun = @gradW_opt_1_fixed_func;
        E = kron(Ik, ones(k));
        [C, R] = gradW_opt_aux(k);
        CRt = R'*C';
        kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));        
        if isempty(varargin)
            varargin = {'relu'}; % varargin = {'pow', 2};
        end
        varargin_grad = [{Ik}, {E}, {CRt}, varargin{:}];
        varargin_reg = [{kE_term}, varargin{:}];

    otherwise
        % correct/default regularizer
        regfun = @regW_fixed;
        gradfun = @gradW_opt_1_fixed;
        E = kron(Ik, ones(k));
        [C, R] = gradW_opt_aux(k);
        CRt = R'*C';
        kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));
        varargin_grad = {Ik, E, CRt};
        varargin_reg = {kE_term};
end

tic;
delta = zeros([d, k]);
for i=1:d
    for j=1:k
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (regfun(Wp, k, s, varargin_reg{:}) - regfun(Wm, k, s, varargin_reg{:}))/(2*e);
    end
end
numgrad = vec(delta);
time_n = toc;

grad_W = gradfun(W, k, s, varargin_grad{:});
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);

if nargout==0
    disp([numgrad grad_W]);
    fprintf('diff grad: %e \n', diff);
end