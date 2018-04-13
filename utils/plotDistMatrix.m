function h = plotDistMatrix(distMat, ind_in, ind_out, nBins, h, lw, fn)% nComp)

if nargin<7, fn = 12; end % fontsize
if nargin<6, lw = 2; end % linewidth
if nargin<5 || isempty(h), h = figure; end; % figure handle
if nargin<4 || isempty(nBins), nBins = 10; end % number of bins

strMethod = {'Inter', 'Intra'};

dist_in = distMat(ind_in); 
dist_in(isinf(dist_in)) = []; % remove inf
dist_out = distMat(ind_out); 
dist_out(isinf(dist_out)) = []; % remove inf

figure(h); hold all; c = lines(2);

[~, x1] = hist(dist_in, nBins);
h1 = ksdensity(dist_in, x1); % Kernel density estimates
f1 = area(x1, h1./sum(h1));
set(f1, 'FaceColor', c(1,:), 'EdgeColor', c(1,:), 'FaceAlpha', 0.3, 'EdgeAlpha',1);%set edge color
f1 = plot(x1, h1./sum(h1), '-','linewidth', lw);

[~, x2] = hist(dist_out, nBins);
h2 = ksdensity(dist_out, x2); % Kernel density estimates
f2 = area(x2, h2./sum(h2));
set(f2, 'FaceColor', c(2,:), 'EdgeColor', c(2,:), 'FaceAlpha', 0.3, 'EdgeAlpha',1);%set edge color
f2 = plot(x2, h2./sum(h2), '-','linewidth', lw);

grid minor; % axis([0 max([x{1}(end),x{2}(end),x{3}(end)]) 0 max([h{3}/sum(h{3}), h{2}/sum(h{2})])]);
% title('Intra-orbit', 'Fontsize', fn); % xlabel('distance', 'Fontsize', fn);
legend([f1 f2], strMethod, 'Fontsize', fn, 'location', 'best');
legend boxoff;
set(gca,'Fontsize', fn);


% hh = figure;
% h1 = histfit(dist_in, nBins, 'kernel'); hold on;
% h2 = histfit(dist_out, nBins, 'kernel');