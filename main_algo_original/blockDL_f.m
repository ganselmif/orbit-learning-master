% Dictionary update (NOT WORKING!)

function D = blockDL_f(MAX_ITER, X, M, D, w, A, lambda1, optimType)

if nargin<8, optimType = 'fminunc'; end

costFunc = @(p) costFunctionD_vec(p, X, M, w, A, lambda1);

vecD = fminWrapper(costFunc, D(:), optimType, MAX_ITER);

k = length(w);
d = length(vecD)/k; 
D = reshape(vecD, d, k); % unroll dictionary variables

%%  project in unit ball the result (seems not to be needed!) 
D  = project_unit(D);