% Gradient of sum regularizerfor multiple orbits (with cross-orbit terms)
%
% Output is not vectorized/Matrix format!

function dW = sgradW_mult_cross(W, setm, k, varargin)

K = unique(setm); % number of orbits in layer

% auxiliary matrices
if nargin<4
    Jk = ones(k);
    R = -Jk + (k)*eye(k);
else
    Jk = varargin{1};
    R = varargin{2};
end

dW = zeros(size(W));
if isa(W,'gpuArray')
    dW = gpuArray(dW);
end
     
for p=K
    ind_p = setm==p;
    Wp = W(:, ind_p);
    Q = 0;
    
    % precomputed quantities
    WpJk = Wp*Jk;
    WpR = Wp*R;
    
    for q=K
        ind_q = setm==q;
        Wq = W(:, ind_q);
        Wqt = Wq';
        Q = Q + Wq*(R*Wqt*WpJk + Jk*Wqt*WpR);
    end
    dW(:, ind_p) = 2*Q;
end

% Q = 0;
% for g=K
%     Wg = W(:, setm==g);
%     Q = Q + Wg*Jk*Wg';
% end
% 
% dW = zeros(size(W));
% for g=K
%     ind_g = setm==g;
%     Wg = W(:, ind_g);
%     WgtWg = (Wg'*Wg);
%     dW(:, ind_g) = 2*(Q-Wg*Jk*Wg')*Wg*R + 2*Wg*(Jk*WgtWg*R + R*WgtWg*Jk);
%     % dW(:, ind_g) = 2*Wg*(Jk*WgtWg*R + R*WgtWg*Jk);
% end

% dW = zeros(size(W));
% for g1=K
%     Dx1 = 0; Dx2 = 0;
%     ind_g = setm==g1;
%     Wp = W(:, ind_g);
%    
%     for g2=K
%         % disp([g1, g2])
%         Wq = W(:, setm==g2);        
%         if g1~=g2
%             Dx1 = Dx1 + Wq*Jk*Wq';
%             Dx2 = Dx2 + Wq*R*Wq';
%         else
%             Ds = 2*Wp*(Jk*(Wp'*Wp)*R + R*(Wp'*Wp)*Jk);
%         end
%         
%     end
%     dW(:, ind_g) = Dx1*Wp*R + Dx2*Wp*Jk + 2*Ds;
% end

% tic;
% t = Jk*Wg'*R;
% t1 = t + t';
% toc
% tic;
% t2 = Jk*Wg'*R + R*Wg'*Jk;
% toc
% sum(sum(t1-t2))


