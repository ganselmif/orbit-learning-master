function y = sthresh(x, sorh, t)
%STHRESH Perform soft or hard thresholding. 
%   Y = STHRESH(X,SORH,T) returns soft (if SORH = 's')
%   or hard (if SORH = 'h') T-thresholding  of the input 
%   vector or matrix X. T is the threshold value.
%
%   Y = STHRESH(X,'s',T) returns Y = SIGN(X).(|X|-T)+, soft 
%   thresholding is shrinkage.
%
%   Y = STHRESH(X,'h',T) returns Y = X.1_(|X|>T), hard
%   thresholding is cruder.

switch sorh
  case 's'   
    tmp = (abs(x)-t);
    tmp = (tmp+abs(tmp))/2;
    y   = sign(x).*tmp;          
    % alternative implementations
    % y = (x-t*x./abs(x)).*(abs(x)>t);
    % y = x.*max(0,1-t./abs(x));
    
  case 'h'
    y   = x.*(abs(x)>t);
 
  otherwise
    error(message('FunctionArgVal:Invalid_ArgVal'))
end

