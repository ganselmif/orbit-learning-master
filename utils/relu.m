% ReLU and leaky ReLU (feedforward computation)

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function y = relu(x, leak_factor)

if nargin==1 || leak_factor == 0 
    y = max(x, 0);
else
    y = x .* (leak_factor + (1 - leak_factor) * (x > 0));
end