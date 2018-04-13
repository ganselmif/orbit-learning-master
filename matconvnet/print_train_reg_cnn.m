% Print comparison plot of loss vs. regularizer
% both are unit norm for relative values

function print_train_reg_cnn(info, regParam)

lw = 2; fn = 10;

figure; clf ;

ob_norm = [info.train.objective];

reg_norm = regParam*[info.train.reg]; % - mean(info.train.reg); 
%reg_norm = reg_norm/norm(reg_norm); % unit norm
loss_norm = [info.train.objective] - regParam*[info.train.reg]; % remove regularizer
% ob_norm = ob_norm - mean(ob_norm); 
%ob_norm = ob_norm/norm(ob_norm); % unit norm

plot(reg_norm, '.-', 'linewidth', lw) ; hold all ;
plot(loss_norm, '.-', 'linewidth', lw) ; 
plot(ob_norm, '.--', 'linewidth', lw) ; 
xlabel('Training epoch'); ylabel('energy') ;
grid on ;
% set(h,'color','none');
title('objetive terms (unit norm)') ;
set(gca,'FontSize', fn);
legend({'reg', 'loss', 'objective'}, 'Fontsize', fn);

% drawnow ;
