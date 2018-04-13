figure;
filename='DicyclicGroup7.txt'
F=Datagen(filename,[1:28]');
d=size(F,2);
r=round(d/4);
for i=1 :d
subplot(r,4,i)
bar(F(:,i)');
title(sprintf('Transformations of [1,2,..,d]%d', i));
end
%round(d/4)