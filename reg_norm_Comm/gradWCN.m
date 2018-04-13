
function dWCN = gradWCN(W, maxp, d, k)

if nargin<3, [d, k] = size(W); end;
if nargin==1, maxp = 5; end;

Jk = ones(k);
Jd = ones(d);
Id = eye(k);

dWCN = zeros([d,k]);

for p=1:maxp
    
    Wp = W.^p;
    %Wp = exp(p*W);
    
    Wpm1 = W.^(p-1);
    %Wpm1 = exp(p*W);    
    N = trace(Wp'*Wp);
    
    %dWCN = project_unit(dWCN + i*((2/N)*(Jd*Wi*2*(k*Id-Jk))-(2/N^2)*trace(Wi*2*(k*Id-Jk)*Wi'*Jd)*Wi).*Wim1);
    dWCN = dWCN + p*((2/N)*(Jd*Wp*2*(k*Id-Jk))-(2/N^2)*trace(Wp*2*(k*Id-Jk)*Wp'*Jd)*Wp).*Wpm1;
end