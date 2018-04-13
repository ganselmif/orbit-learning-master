function [X] = Datagen(filename, t)

% t = 100;
P = Pmat(filename);
n = size(P(:,:,1),1);

%X0 = 0.5 + 1*randn(n,t);
X0 = rand(n,t);

order = size(P, 3);
e = size(X0,2);

for i=1:order
    X(:,:,i)= P(:,:,i)*X0;
end
% X = reshape(X, [n, order*e]);
X = permute(X, [3, 1, 2]);
X = reshape(X, [n, order*e]);
end

%generate from a set of input vectors X0 all their transformations w.r.t
%the group generators