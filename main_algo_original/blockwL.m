% Update of the w vector
% Note: (with sparsity, may be not necessary)
%
% tau: the proximal gradient constant
% beta: the proximal gradient annealing constant

function w = blockwL(MAX_ITER, tau, beta, M, D, w, lambda1)

% if nargin<8
%     gamma = 0.1;
% end

% cached computations: compute once
DtD = D'*D;
k = length(w);
Mt = permute(M, [2 1 3]);
MplusMt = M + Mt; % M + M'

for iterNo = 1:MAX_ITER
    
    for indM = 1:k % loop over vector dimension
        
        % grad_wi = 2*lambda1*(trace(M(:,:,indM)*sum(Dw1(w,Mt,k),3)) - trace(DtD*MplusMt(:,:,indM)));
        grad_w(indM, 1) = lambda1*(trace(M(:,:,indM)*sum(Dw1(w, Mt, k),3)) - 0.5*trace(DtD*MplusMt(:,:,indM)));
        %w(indM) = w(indM) - tau*grad_w(indM);
        
        % why sparsity here?
        % w(indM) = sthresh(w(indM) - lambda*grad_wi, 's', lambda*gamma);
        
    end
    w = w - tau*grad_w;
    tau = beta*tau;
    
end
w  = project_unit(w); % w = w./norm(w);