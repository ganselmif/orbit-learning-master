function [X] = DatagenS(filename,t);
P = Pmat(filename);
order=size(P,3);
n = size(P(:,:,1),1);
X0=rand(n,1)*ones(t, 1)' + randArrayInRange([n,t], 0, 1);
e=size(X0,2);
for i=1: order
    X(:,:,i)= P(:,:,i)*X0;
end
X=permute(X,[3,1,2]);
X=reshape(X,[n,order*e]);
end

%generate from a set of input vectors X0 all their transformations w.r.t
%the group generators