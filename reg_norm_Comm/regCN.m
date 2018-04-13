function regcomN = regCN(W, maxp, d, k)

Jk = ones(k);
Jd = ones(d);
Id = eye(k);

regcomN = 0;
for p=1:maxp
    
    Wp = W.^p;
    %Wp = exp(p*W);
    
    N = trace(Wp'*Wp);
    regcomN = regcomN + (2/N)*trace(Wp*(k*Id-Jk)*Wp'*Jd);    
end