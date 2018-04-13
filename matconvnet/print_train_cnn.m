% Print summary plots for training a matconvnet network

function print_train_cnn (info)

lw = 2; fn = 10;

figure; clf;

subplot(1,2,1) ;
semilogy([info.train.objective], '.-', 'linewidth', lw); hold all ;
semilogy([info.val.objective], '.-', 'linewidth', lw); hold all ;
xlabel('Training epoch'); ylabel('energy');
grid on;
% set(h,'color','none');
title('objective');
set(gca,'FontSize', fn); 
legend({'train', 'val'}, 'Fontsize', fn);

subplot(1,2,2) ;
plot([info.train.top1err], '.-', 'linewidth', lw); hold all;
plot([info.val.top1err], '.-', 'linewidth', lw); hold all;
grid on;
xlabel('Training epoch'); ylabel('error');
% set(h,'color','none') ;
title('top1error');
set(gca,'FontSize', fn); 
legend({'train', 'val'}, 'Fontsize', fn);

% drawnow ;
