% Optimized getT computation

function T = genT_opt(k)
k2 = k^2;
T = zeros([k2, k2]);

for i=1:k2
    i_ind = 1+k*(i-1)-(k2-1)*floor((i-1)/k);
    T(i, i_ind) = 1;
end

% for i=1:k^2
%     for j=1:k^2
%         if j==1+k*(i-1)-(k^2-1)*floor((i-1)/k);
%             T(i,j) = 1;
%         else
%             T(i,j) = 0;
%         end
%     end
% end
    