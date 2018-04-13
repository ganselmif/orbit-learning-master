% Check analytic vs. numerical gradient for (multiple) orbit regularizer
%
% % Example (how to test):
% 
% k = 20; Wo = rand(k, k); % need to use random matrix! 
% tic; testgradW(Wo, s, 1); toc 

function [diff, time_n] = testgradW_mult(W, setm, s, formReg)
e = 10^-8;

% setm = kron
if nargin<4, formReg = 1; end

[d, K] = size(W);
k = K/length(unique(setm)); % number of groups/orbits

% Type of regularizer (and gradient to check)
switch formReg
    case 1
        % All-differences reg., multiple orbits
        regfun = @regW_mult;
        gradfun = @gradW_opt_1_mult;
        
        % auxiliary matrices
        E = kron(eye(k), ones(k));
        varargin_reg = {s, E};
        
    case 2
        % Sum reg., multiple orbits, cross
        regfun = @sregW_mult_cross;
        gradfun = @sgradW_mult_cross; 
        
        % auxiliary matrices
        Jk = ones(k);
        R = -Jk + (k)*eye(k);
        varargin_reg = {Jk, R};
        varargin_grad = {Jk, R};

    case 3
        % All-differences reg., multiple orbits, cross
        regfun = @regW_mult_cross;
        gradfun = @gradW_opt_1_mult_cross;       
        
        % auxiliary matrices
        Ik = sparse(eye(k)); %Ik = eye(k);
        E = kron(Ik, ones(k));
        [C, R] = gradW_opt_aux(k);
        % S = genS(k);
        % SCt = S*C';
        % STCt = genT_opt(k)*S*C';        
        Ct = C';
        CRt = R'*Ct;
        CTt = genT_opt(k)*Ct;
        varargin_reg = {s, E};
        varargin_grad = {s, Ik, E, CRt, Ct, CTt};
        
end

tic;
delta = zeros([d, K]);
for i=1:d
    for j=1:K
        Wp = W; Wp(i,j) = Wp(i,j) + e;
        Wm = W; Wm(i,j) = Wm(i,j) - e;
        delta(i,j) = (regfun(Wp, setm, k, varargin_reg{:}) - regfun(Wm, setm, k, varargin_reg{:}))/(2*e);
    end
end
numgrad = vec(delta);
time_n = toc;

switch formReg
    case 1
        grad_W = gradfun(W, setm, k, s);
    otherwise
        grad_W = gradfun(W, setm, k, varargin_grad{:});
        grad_W = grad_W(:); % vectorize
end

disp([numgrad grad_W]);
diff = norm(numgrad - grad_W)/norm(numgrad + grad_W);
fprintf('w gradient: %e \n', diff);
