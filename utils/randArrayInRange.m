% Generate array of narrays each of size arraysize filled with random
% numbers in the interval [a,b]: used for inizialization of M,A,w

function result = randArrayInRange(arraysize, a, b)
    
result = a + (b-a).*rand(arraysize);