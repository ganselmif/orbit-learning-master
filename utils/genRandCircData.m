% Generate concatenations of circular data for testing

function X = genRandCircData(d, N, opt, sd)

if nargin<4, sd = 0; end
if nargin<3, opt = 'other'; end
if nargin<2, N = 1; end

rng(sd, 'twister'); % Fix seed for reproducible results

switch opt
    
    case 'random'
        
        a = 0; b = 1; % range of values
        X = [];
        for n = 1:N
            r = round(a + (b-a).*rand(d,1), 3); % round symbols to second decimal
            X = [X gallery('circul', r)];
        end
        
    otherwise
        
        X = [];
        for n = 0:N-1
            % c = linspace(n, n+1, d);
            r = n + rand(d,1);
            X = [X gallery('circul', r)];
        end
        X = X./N;
end
