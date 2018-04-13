% Sum regularizer for multiple orbits (using also cross terms)

function reg = sregW_mult_cross(W, setm, k, varargin)

K = unique(setm); % number of orbits in layer

% auxiliary matrices
if nargin<4
    Jk = ones(k);
    R = -Jk + (k)*eye(k);
else
    Jk = varargin{1};
    R = varargin{2};
end

reg = 0;
for g1=K
    W1 = W(:, setm==g1);
    W1RW1t = W1*R*W1';
    for g2=K
        W2 = W(:, setm==g2);
        reg = reg + trace(W1RW1t*(W2*Jk*W2'));
    end
end