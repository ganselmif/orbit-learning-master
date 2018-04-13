function W = project_pos(W)
% PROJECT_POS Project a point onto the nonnegative orthant.
%
%   project_pos(o) is the positive part of o.

W = max(W, 0);
