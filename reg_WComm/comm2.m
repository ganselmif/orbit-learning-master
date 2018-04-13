% Implements comm(comm(A, B), B) for faster computations

function commu = comm2(A, B)
AB = A*B;
ABt = AB';
% commu = AB*B - 2*B*AB + B*B*A;
commu = AB*B - 2*ABt*B + B*ABt;