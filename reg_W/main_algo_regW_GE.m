clear; close all;
testCase = 'group'; 

switch testCase
    case 'group'
        
        groupnametxt = 'DihedralGroup6.txt';
        
        %P = Pmat(groupnametxt);
        %W = DatagenS(groupnametxt,1);
        
        Wg = genGroupData(groupnametxt, 1);
        %W = DatagenS(groupnametxt,1) + 0.5*randArrayInRange([d, k, 1], a, b);
        [d, k] = size(Wg);
        
        Wo = Wg + 0.3*rand(size(W));
        
    otherwise
        
        d = 256; % 15^2;
        k = 10; %10; % 8;
        
        a = 0;
        b = 1;
        Wo = randArrayInRange([d, k, 1], a, b);
end

E = kron(eye(k), ones(k));
Ik = sparse(eye(k));

[C, R] = gradW_opt_aux(k);
CRt = R'*C';

ITER = 800; % num of iterations
lambda = 1;
s_vals = [0.01 0.1 1:1:10 20 30];
% s = 2;
clear r;
for i=1:length(s_vals)
    clear reg nder
    W = Wo;
    s = s_vals(i);
    for iterNo = 1 : ITER
        
        grad_W = reshape(gradW_opt_1_fixed(W, k, s, Ik, E, CRt), [d, k]) ;
        W = W - lambda*grad_W;
        
        reg(iterNo) = regW_fixed(W, k, s, E);
        % hist(reg_c(:), 20); % drawnow;
        nder(iterNo) = norm(vec(grad_W));
    end
    r(i) = reg(1)/reg(end);
end
figure; plot(s_vals, r, 'o-', 'linewidth', 2);
ylabel('reg(1)/reg(end)'); xlabel('s'); title(sprintf('d = %d, k = %d', d, k));


s = 0.01; W = Wo; clear reg nder r
for iterNo = 1:ITER
    
    grad_W = reshape(gradW_opt_1_fixed(W, k, s, Ik, E, CRt),[d,k]) ;
    W = W - lambda*grad_W;
    
    reg(iterNo)= regW_fixed(W, k, s, E);
    nder(iterNo) = norm(vec(grad_W));    

end

%% Plots and Figures

figure;
r = reg(1)/reg(end);
subplot(2,2,1); imagesc(W); colorbar; title('Weights')
subplot(2,2,2); imagesc(W'*W); colorbar; title('Gramian')
subplot(2,2,3); plot(1:ITER, reg(1:end), '.-'); axis tight; title(sprintf('reg(1)/reg(end)= %e ',r));
subplot(2,2,4); plot(1:ITER, nder(1:end), '.-'); axis tight; title('norm dW');

