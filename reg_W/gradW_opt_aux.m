% Auxiliary functions for optimized computation for gradW

function [C, R] = gradW_opt_aux(k)

R = sparse(eye(k^2) + genT_opt(k));
C = genC_opt(k);