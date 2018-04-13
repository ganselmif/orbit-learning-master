% Computes N x N Gramian matrix G of observations in d x N matrix W
%
% W is arranged columnwise
%
% Inner products can be:
% - in input space, linear (standard Gramian)
% - in chosen feature map, non-linear (by applying pointwise nonlinearity


function [G, fW, dfdW] = gramMat(W, typeMap, p)

if nargin<2 || isempty(typeMap), typeMap = 'linear'; end % defaults to linear

%% choice of functions to apply to W (elementwise)
switch typeMap
    
    case 'pow';
        if nargin<3, p = 3; end
        fW = W.^p;
        
    case 'sin';
        if nargin<3, p = 1; end
        fW = sin(p*W);
        
    case 'relu'
        if nargin<3, p = 0; end
        fW = relu(W - p);
        
    otherwise % 'linear'
        fW = W;
end

%% Gramian computation
G = fW'*fW;

%% column normalization operation (?)
% normMat = diag(1./sqrt(sum(fW.^2)));
% G = normMat'*G*normMat; 
% % fW = fW*normMat; G = fW'*fW;
% if nargout>2
%     fW = fW*normMat;
% end

%% Gradient of Grammian wrt. W
if nargout>2
    switch typeMap
        case 'pow';
            dfdW = p*W.^(p-1);
            
        case 'sin';
            dfdW = p*cos(p*W);
            
        case 'relu'
            dfdW = (W > p);
            
        otherwise % 'linear'
            dfdW = ones(size(fW));
    end
end


    