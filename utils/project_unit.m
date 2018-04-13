function D  = project_unit(D)
% PROJECT_UNIT Project a dictionary (matrix columns) onto the unit ball.

D = bsxfun(@rdivide, D, max(sqrt(sum(D.^2)),1));

