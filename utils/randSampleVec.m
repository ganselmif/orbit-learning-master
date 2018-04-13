% Sampling random or pseudo-random vectors 
% 
% - uniformly
% - normally
% - uniformly in unit ball


function T = randSampleVec(d, N, samplType, C)

if nargin<3, samplType = 'usimp'; end

switch samplType
    
    case 'nsimp'
        %% Normal-distributed
        T = randn(d, N); 
    
    case 'usimp'
        %% Vectors in [0, 1]^d simplex
        T = rand(d, N); % random/uniformly in [0,1] templates
        
    case 'uball'
        %% Vectors uniformly distributed in unit-ball
        T = randsphere(N, d, 1)';
        
    case 'mball'
        %% Vectors in multiple balls (each one unifrormly in unit ball).
        % C subsequent points are sampled uniformly with radius s from each smaller ball
        To = randsphere(C, d, 1)'; % Pool of generating (seed) points        
        s = 0.2; T = [];
        for c = 1:C
            T = [T bsxfun(@plus, randsphere(N/C, d, s)', To(:, c))]; % Seed point perturbed by vector with std s
        end    
end