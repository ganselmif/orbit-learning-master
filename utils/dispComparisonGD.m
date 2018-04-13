% Display ComparisonGD results
%
% see also ComparisonGD.m

% GE, CBMM/LCSL/MIT, gevang@mit.edu

function [stats_ds_in, stats_ds_out] = dispComparisonGD(ds_in, ds_out, nBins, strMethod)

if nargin<=3, nBins = 40; end
nMethods = size(ds_in, 2);

if nargin<4,
    strMethod = {'sym', 'regw', 'rand'};
    if nMethods == 4;
        strMethod{4} = 'dif-sym';
    end
end
% dsoh = [ds_in(:,1);ds_out(:,1)];

disp('---------- Within orbits (signature distance)');
md_ds_in =  median(ds_in);
mn_ds_in =  mean(ds_in);
sd_ds_in =  std(ds_in);
for i=1:nMethods
    fprintf('%7s (med, mean, std) = (%f, %f, %f)\n', strMethod{i}, md_ds_in(:,i), mn_ds_in(:,i), sd_ds_in(:,i));
end

disp('---------- Across orbits (signature distance)');
md_ds_out =  median(ds_out);
mn_ds_out =  mean(ds_out);
sd_ds_out =  std(ds_out);
for i=1:nMethods    
    fprintf('%7s (med, mean, std) = (%f, %f, %f)\n', strMethod{i}, md_ds_out(:,i), mn_ds_out(:,i), sd_ds_out(:,i));
end

if nargout~=0
    stats_ds_in = [md_ds_in; mn_ds_in; sd_ds_in];
    stats_ds_out = [md_ds_out; mn_ds_out; sd_ds_out];
end

%%  Within orbit distance histograms
h1 = figure; hold all; c = lines(nMethods); lw = 2; fn =12;
for i=1:nMethods
    [h{i}, x{i}] = hist(ds_in(:,i), nBins);
    f{i} = area(x{i}, h{i}./sum(h{i}));
    set(f{i}, 'FaceColor', c(i,:), 'EdgeColor', c(i,:), 'FaceAlpha', 0.3,'EdgeAlpha',1);%set edge color
    f{i} = plot(x{i}, h{i}./sum(h{i}), '-','linewidth', lw);
    
    max_x(i) = max(x{i}); max_y(i) = max(h{i}./sum(h{i}));
end
max_x_h1 = max(max_x); max_y_h1 = max(max_y); clear max_x max_y

grid minor; axis tight; % axis([0 max([x{1}(end),x{2}(end),x{3}(end)]) 0 max([h{3}/sum(h{3}), h{2}/sum(h{2})])]);
title('Intra-orbit', 'Fontsize', fn); xlabel('distance', 'Fontsize', fn); 
legend([f{:}], strMethod, 'Fontsize', fn, 'location', 'best');
set(gca,'Fontsize', fn);


%% Across orbit distance histograms
h2 = figure; hold all; c = lines(nMethods); % lw = 2; fn =12;
for i=1:nMethods
    [h{i}, x{i}] = hist(ds_out(:,i), nBins);    
    f{i} = area(x{i}, h{i}./sum(h{i}));
    set(f{i}, 'FaceColor', c(i,:), 'EdgeColor', c(i,:), 'FaceAlpha', 0.3,'EdgeAlpha',1);%set edge color
    f{i} = plot(x{i}, h{i}./sum(h{i}), '-','linewidth', 2);
    
    max_x(i) = max(x{i}); max_y(i) = max(h{i}./sum(h{i}));
end
max_x_h2 = max(max_x); max_y_h2 = max(max_y);  clear max_x max_y

grid minor; axis tight;
title('Inter-orbit', 'Fontsize', fn); xlabel('distance', 'Fontsize', fn); 
legend([f{:}], strMethod, 'Fontsize', fn, 'location', 'best');
set(gca,'Fontsize', fn);


%% Set same axis to both for direct comparisons
x_lim = [0 max(max_x_h1, max_x_h2)]; y_lim = [0 max(max_y_h1, max_y_h2)];
set(h1.CurrentAxes, 'XLim', x_lim, 'YLim', y_lim);
set(h2.CurrentAxes, 'XLim', x_lim, 'YLim', y_lim);


% figure; hold all; c = lines(3);
% x_max = max(ds_in(:)); 
% for i=1:nMethods 
%     [~, h{i}, x{i}] = kde(ds_in(:,i), nBins, -x_max, x_max);
%     % [dP,xgridP] = ksdensity(scoreP,'npoints',N,'function','pdf');    
%     f{i} = plotdens1D(h{i}./sum(h{i}), x{i}, ds_in(:,i), [], 2, c(i,:));
% end
% x_axis_max = max([x{1}(end),x{2}(end),x{3}(end)]);
% y_axis_max = max([h{3}/sum(h{3}); h{2}/sum(h{2})]);
% grid minor; axis([0 x_axis_max 0 y_axis_max]);
% title('Within Orbits'); xlabel('distance'); 
% legend([f{1}, f{2}, f{3}], strMethod);

%     [h1, x1] = hist(dsoh(:), 40);
%     [h2, x2] = hist(dsod(:), 40);
%     [h3, x3] = hist(dsor(:), 40);
%     figure; plot(x1, h1./sum(h1), x2, h2/sum(h2), x3, h3/sum(h3), '-','linewidth', 2);
%     grid minor; axis tight; title('all'); legend('orbit', 'regw', 'random');
%     figure;
%     subplot 131; plot(x1, h1, '.-', 'linewidth', 2); title('orbit');
%     subplot 132; plot(x2, h2, 'r.-','linewidth', 2); title('regw');
%     subplot 133; plot(x3, h3, 'k.-','linewidth', 2); title('random');
%     figure;
%     subplot 131; bar(x1, h1); title('orbit');
%     subplot 132; bar(x2, h2, 'r'); title('weights');
%     subplot 133; bar(x3, h3, 'k'); title('random');
