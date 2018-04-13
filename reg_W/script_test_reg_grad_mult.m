% Check multiple groups/orbits 
clear;

filename = 'CyclicGroup6.txt';
% W = DatagenS(filename,1);
M = 1;
W = genGroupData(filename, M);
[d, K] = size(W);

k = K/M; 
setm = kron(1:M, ones(1, k));

%% Noisy tests
% Test 1: AWGN noise  
W1 = W + 0.001*randn([d, K]);

% Test 2: noise
W2 = rand([d, K]);


%% ALL-DIFFERENCES REGULARIZER (Multiple Orbits) 
%% Regularizer values 
s = 0.001;
regW_mult(W, setm, k, s)
regW_mult(W1, setm, k, s)
regW_mult(W2, setm, k, s)

%% Gradients and comparisons
testgradW_mult(W, setm, s, 1);
testgradW_mult(W1, setm, s, 1);
testgradW_mult(W2, setm, s, 1);
%%*************************************************************************

%% SUM REGULARIZER (Multiple Orbits, Cross terms) 

sregW_mult_cross(W, setm, k)
sregW_mult_cross(W1, setm, k)
sregW_mult_cross(W2, setm, k)

%% Speed
if 0
    M = 100; filename = 'CyclicGroup31.txt';
    W = genGroupData(filename, M);
    [d, K] = size(W); k = K/M;
    setm = kron(1:M, ones(1, k));
    W1 = W + 0.1*randn([d, K]);
    tic; regW_mult(W1, setm, k, s); toc
    tic; sregW_mult_cross(W1, setm, k); toc
end

%dW = sgradW_mult_cross(W, setm, k);

%% Check gradient
testgradW_mult(W, setm, [], 2);
testgradW_mult(W1, setm, [], 2);
testgradW_mult(W2, setm, [], 2);
%%*************************************************************************

%% ALL-DIFFERENCES REGULARIZER (Multiple Orbits, Cross terms) 
s = 0.001;
regW_mult_cross(W, setm, k, s)
regW_mult_cross(W1, setm, k, s)
regW_mult_cross(W2, setm, k, s)

%dW = gradW_opt_1_mult_cross(W, setm, k, s);
testgradW_mult(W, setm, s, 3);
testgradW_mult(W1, setm, s, 3);
testgradW_mult(W2, setm, s, 3);




