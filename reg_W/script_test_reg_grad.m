clear;

%% Group matrix
filename = 'CyclicGroup6.txt';
W(:,:,1) = genGroupData(filename, 1);
[d, k] = size(W(:,:,1));

%% Noisy tests
% Test 2: AWGN noise
W(:,:,2) = W(:,:,1) + 0.5*randn([d, k]);
% Test 3: noise
W(:,:,3) = rand([d, k]);
% Test 4: 1 element
W(:,:,4) = W(:,:,1); W(randi(d), randi(k), 4) = mean(mean(W(:,:,1)));

nW = size(W, 3); % number of tests

%% Regularizer values
s = 0.01;

E = kron(eye(k), ones(k));
kE_term = (k*E - 1 - 0.5*(k-1)*eye(k^2));

for i=1:nW
    reg = regW_fixed(W(:,:,i), k, s, kE_term);
    disp(reg);
end

%% (Square) Regularizer values 
% s = 0.001;
% for i=1:4, regW_2(W(:,:,i), k, s, kE_term), end

%% Gradients compare*******************************************************
s = 0.01;
for i=1:nW
    diff = testgradW(W(:,:,i), s, 5);
    disp(diff);
end


%% ************************************************************************
%% Gramian with (nonlinear) feature maps
% argvar0 = 'linear';
argvar1 = 'relu'; % relu 
argvar2 = {'pow', 4}; % power 
argvar3 = {'sin', 0.5}; % sin

for i=1:nW
    reg = regW_fixed_func(W(:,:,i), k, s, kE_term, argvar1); 
    disp(reg);
end

for i=1:nW
    diff = testgradW(W(:,:,i), s, 6, argvar1);
    disp(diff);
end


if 0
    %% ************************************************************************
    %% Non-zero
    s = 0.01;
    for i=1:nW
        diff = regW_nz(W(:,:,i), k, s);
        disp(diff);
    end
    
    for i=1:nW
        diff = testgradW(W(:,:,i), s, 3);
        disp(diff);
    end
    
    %% ************************************************************************
    s = 0.001; %eps;
    for i=1:nW
        diff = testgradW(W(:,:,i), s, 4);
        disp(diff);
    end
    
    
    %% ************************************************************************
    %% Gradient scaling
    
    % %% C and T computations
    % profile on
    % k = 25;
    % C5 = genC_opt(k);
    % profile viewer
    % % T = genT(k);
    %
    % k = 25;
    % tic; C = genC(k); toc
    % %tic; C1 = genC1(k); toc
    % %tic; C2 = genC2(k); toc
    % %tic; C3 = genC3(k); toc
    % %tic; C4 = genC4(k); toc
    % tic; C5 = genC_opt(k); toc
    % %sum(sum(C-C1))
    % %sum(sum(C-C2))
    % %sum(sum(C-C3))
    % %sum(sum(C-C4))
    % sum(sum(C-C5))
    
    
    %% Gradient scaling/computation
    
    % k = 25; Wo = rand(k,k);
    % tic; dW = gradW(Wo, k, s); toc
    % tic; dW1 = gradW_opt(Wo, k, s); toc
    % sum(sum(dW-dW1))
    % % disp([dW, dW1]);
    %
    % profile on
    % dW = gradW_opt(Wo, k, s);
    % profile viewer
    
    %% Dependence on K plots
    % time_k(1) = nan;
    % diff_k(1) = nan;
    clear tim_k diff_k time_num_k
    k_max = 30;
    for k=2:k_max
        Wo = rand(k, k); d = k;
        % tic; dW = gradW(Wo, k, s); time_k_old(k) = toc;
        tic; dW = gradW_opt_1_fixed(Wo, k, s); time_k(k) = toc;
        [diff_k(k), time_num_k(k)] = testgradW(Wo, s, 5);
    end
    
    %% Comparison figures
    dirFigs = '/media/gevang/Data/work/exp/orblearn/figs';
    
    % Time comparisons
    figure; hold all;
    k_min_show = 15;
    plot(k_min_show:k_max, time_k(k_min_show:end), 'o-', 'linewidth', 2);
    plot(k_min_show:k_max, time_num_k(k_min_show:end), 'o-', 'linewidth', 2);
    % plot(k_min_show:k_max, time_k_old(k_min_show:end), 'o-', 'linewidth', 2);
    axis tight; grid on;
    xlabel('k'); ylabel('time (sec)');
    set(gca, 'fontsize', 12);
    legend('Analytic', 'Numerical', 'location', 'NorthWest');
    
    filename = fullfile(dirFigs, 'grad_time_compare');
    printif(gcf, filename, true)
    
    % New gradW times only
    figure; hold all;
    k_min_show = 2;
    plot(k_min_show:k_max, time_k(k_min_show:end), 'o-', 'linewidth', 2);
    axis tight; grid on;
    xlabel('k'); ylabel('time (sec)');
    set(gca, 'fontsize', 12);
    
    filename = fullfile(dirFigs, 'grad_time');
    printif(gcf, filename, true)
end
