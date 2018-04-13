% Compute the sum of scaled matrices, i.e. \sum_i(w(i)*Mi) 

function S = Dw1(w, M, k, opt)

if nargin<4, opt = 'tens'; end
if nargin<3, k = length(w); end

switch opt
    
    case 'loop'
        
        S = zeros(size(M));
        for i = 1:k
            S(:,:,i) = w(i)*M(:,:,i);
        end
        
    otherwise
        
        b(1, 1, :) = w;      % create 3D tensor with scalar flats
        S = repmat(b, k).*M; % elementwise multiplication
end

%calculating sum_{i}w_{i}M_{i}